fn main() {
    // Ask pyo3 for the linker flags an extension module needs on this
    // platform. The one that matters most is macOS's
    // `-undefined dynamic_lookup`, which defers interpreter symbols (such
    // as `Py_True`) to load time; skipping it makes a bare `cargo build`
    // die at link time with "symbol(s) not found for architecture arm64".
    // maturin injects the same flags itself — this call is what keeps a
    // plain `cargo build` of the workspace working as well.
    pyo3_build_config::add_extension_module_link_args();
}
