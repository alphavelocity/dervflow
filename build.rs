// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[cfg(feature = "python")]
fn main() {
    let config = pyo3_build_config::get();

    for line in config.build_script_outputs() {
        println!("{line}");
    }

    if !config.suppress_build_script_link_lines {
        if let Some(lib_dir) = &config.lib_dir {
            println!("cargo:rustc-link-search=native={}", lib_dir);
        }
        if let Some(lib_name) = &config.lib_name {
            println!("cargo:rustc-link-lib={}", lib_name);
        }
    }

    for extra in &config.extra_build_script_lines {
        println!("{extra}");
    }

    if let Some(prefix) = &config.python_framework_prefix {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{prefix}");
    }

    pyo3_build_config::add_extension_module_link_args();
}

#[cfg(not(feature = "python"))]
fn main() {}
