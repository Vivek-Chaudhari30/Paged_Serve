"""Tests for the optional CUDA extension and its loader.

Two audiences. The unmarked tests assert that a *missing* extension is a
supported state — they run on a laptop and are the ones that keep
``import pagedserve`` working without a CUDA toolkit. The ``cuda_ext`` tests
assert the compiled module actually works, and are skipped everywhere it has
not been built.
"""

from __future__ import annotations

import pytest

from pagedserve import extension

torch = pytest.importorskip("torch")


def has_extension() -> bool:
    return extension.is_available()


requires_extension = pytest.mark.skipif(
    not has_extension(),
    reason=(
        "CUDA extension not built. This is a SKIP, not a pass: run "
        "`python setup.py build_ext --inplace` on a CUDA machine to exercise it."
    ),
)


class TestOptionalImport:
    def test_importing_the_package_never_needs_the_extension(self):
        """AGENTS.md §4.3: `import pagedserve` must work with no nvcc.

        This is what lets the whole project be developed on a laptop.
        """
        import importlib

        import pagedserve

        importlib.reload(pagedserve)
        assert pagedserve.__version__

    def test_load_never_raises(self):
        """A missing extension is an expected state, not an error.

        The fallback to the gather backend is a supported configuration, so a
        loader that raised would turn a normal laptop into a broken one.
        """
        assert extension.load() is None or extension.load() is not None

    def test_is_available_agrees_with_load(self):
        assert extension.is_available() == (extension.load() is not None)

    def test_the_reason_is_actionable_when_absent(self):
        reason = extension.unavailable_reason()
        if extension.is_available():
            assert reason is None
        else:
            # An error that does not say what to do next is a dead end.
            assert "build_ext" in reason

    def test_loads_without_torch_already_imported(self):
        """The extension must not depend on someone else importing torch first.

        It links against libc10 and libtorch, which live in torch's package
        directory rather than on the loader's default search path. A caller that
        has already imported torch resolves them by accident; a fresh process
        that has not gets "libc10.so: cannot open shared object file" and a
        perfectly good build looks broken. A test suite hides this, because
        importing torch is the first thing most test modules do.
        """
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                # Distinct tokens, not AVAILABLE/UNAVAILABLE: the second
                # contains the first as a substring.
                "from pagedserve.extension import is_available;"
                "print('EXT=YES' if is_available() else 'EXT=NO')",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        verdict = "EXT=YES" in result.stdout
        # In a fresh process the answer must match this one. Whether the
        # extension exists is environment-dependent; agreeing is not.
        assert verdict == extension.is_available(), (
            f"a fresh process disagrees about the extension.\n"
            f"  here: {extension.is_available()}\n"
            f"  fresh: {result.stdout.strip()}\n{result.stderr[-500:]}"
        )

    def test_repeated_calls_are_cached(self):
        first = extension.load()
        second = extension.load()
        assert first is second


@pytest.mark.cuda_ext
@requires_extension
class TestBuildCanary:
    """Does the toolchain, the binding, and the tensor round-trip work?

    Deliberately separate from any attention math. When the real kernel breaks,
    these answer "is it the environment or is it me?" in one command.
    """

    def test_add_one_round_trips_a_tensor(self):
        module = extension.load()
        source = torch.arange(8, dtype=torch.float32, device="cuda")
        result = module.add_one(source)
        assert torch.equal(result, source + 1)

    def test_does_not_mutate_its_input(self):
        module = extension.load()
        source = torch.zeros(4, dtype=torch.float32, device="cuda")
        module.add_one(source)
        assert torch.equal(source, torch.zeros(4, dtype=torch.float32, device="cuda"))

    def test_handles_a_non_contiguous_input(self):
        """Flat indexing on a strided tensor reads wrong elements silently."""
        module = extension.load()
        base = torch.arange(16, dtype=torch.float32, device="cuda").view(4, 4)
        strided = base[:, ::2]
        assert not strided.is_contiguous()
        assert torch.equal(module.add_one(strided), strided + 1)

    def test_handles_an_empty_tensor(self):
        module = extension.load()
        empty = torch.empty(0, dtype=torch.float32, device="cuda")
        assert module.add_one(empty).numel() == 0

    def test_large_input_crosses_many_blocks(self):
        module = extension.load()
        source = torch.randn(1_000_003, dtype=torch.float32, device="cuda")
        assert torch.allclose(module.add_one(source), source + 1)

    def test_rejects_a_cpu_tensor(self):
        module = extension.load()
        with pytest.raises(RuntimeError, match="CUDA tensor"):
            module.add_one(torch.zeros(4, dtype=torch.float32))

    def test_rejects_the_wrong_dtype(self):
        """Reading float32 out of a float16 buffer would return garbage."""
        module = extension.load()
        with pytest.raises(RuntimeError, match="float32"):
            module.add_one(torch.zeros(4, dtype=torch.float16, device="cuda"))
