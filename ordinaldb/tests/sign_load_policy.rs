use ordinaldb::artifacts::MANIFEST_FILE;
use ordinaldb::manifest::{CreateManifestOptions, VerifyOptions};
use ordinaldb::{
    BuildOptions, DenseError, DenseLoadOptions, OrdinalIndex, SignLoadPolicy, SignPolicy,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * dim];
    for row in 0..n {
        for col in 0..dim {
            let x = (((row + 3) * (col + 5) + row * 17 + col * 11) % 37) as f32 - 18.0;
            out[row * dim + col] = x / 19.0;
        }
    }
    out
}

fn temp_path(name: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{name}-{}-{stamp}", std::process::id()))
}

fn cleanup(path: &Path) {
    let _ = fs::remove_dir_all(path);
    let _ = fs::remove_file(path);
}

fn write_bundle(name: &str, dim: usize, sign: SignPolicy) -> PathBuf {
    let mut idx =
        OrdinalIndex::new_with_build_options(dim, 2, BuildOptions { sign }).expect("construction");
    idx.add_2d(&vectors(4, dim), dim).expect("add batch");
    let bundle = temp_path(name);
    cleanup(&bundle);
    idx.write_verified_bundle(&bundle, CreateManifestOptions::default(), Vec::new())
        .expect("write bundle");
    bundle
}

fn open_with(bundle: &Path, sign: SignLoadPolicy) -> Result<OrdinalIndex, DenseError> {
    OrdinalIndex::open_verified(
        bundle.join(MANIFEST_FILE),
        VerifyOptions::default(),
        DenseLoadOptions {
            sign,
            ..DenseLoadOptions::default()
        },
    )
}

fn unwrap_err(result: Result<OrdinalIndex, DenseError>, context: &str) -> DenseError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(err) => err,
    }
}

#[test]
fn default_load_options_require_sign_if_supported() {
    assert_eq!(
        DenseLoadOptions::default(),
        DenseLoadOptions {
            sign: SignLoadPolicy::RequireIfSupported,
            expected_dim: None,
            expected_bits: None,
        }
    );
}

#[test]
fn default_load_rejects_sign_capable_bundle_without_sidecar() {
    let bundle = write_bundle(
        "ordinaldb-sign-load-default-rejects.odb",
        64,
        SignPolicy::Disabled,
    );
    let err = unwrap_err(
        OrdinalIndex::open_verified(
            bundle.join(MANIFEST_FILE),
            VerifyOptions::default(),
            DenseLoadOptions::default(),
        ),
        "sign-capable bundle without sidecar must fail the default load",
    );
    assert!(matches!(err, DenseError::MissingSignSidecar), "{err}");
    cleanup(&bundle);
}

#[test]
fn any_loads_sign_capable_bundle_without_sidecar() {
    let bundle = write_bundle("ordinaldb-sign-load-any.odb", 64, SignPolicy::Disabled);
    let loaded = open_with(&bundle, SignLoadPolicy::Any).expect("Any tolerates a missing sidecar");
    assert!(!loaded.has_sign_sidecar());
    assert_eq!(loaded.len(), 4);
    cleanup(&bundle);
}

#[test]
fn forbid_rejects_signed_bundle() {
    let bundle = write_bundle("ordinaldb-sign-load-forbid.odb", 64, SignPolicy::Optional);
    let err = unwrap_err(
        open_with(&bundle, SignLoadPolicy::Forbid),
        "Forbid must reject a bundle that declares a sidecar",
    );
    assert!(matches!(err, DenseError::SignSidecarForbidden), "{err}");
    cleanup(&bundle);
}

#[test]
fn forbid_loads_unsigned_bundle() {
    let bundle = write_bundle(
        "ordinaldb-sign-load-forbid-unsigned.odb",
        64,
        SignPolicy::Disabled,
    );
    let loaded = open_with(&bundle, SignLoadPolicy::Forbid).expect("no sidecar to forbid");
    assert!(!loaded.has_sign_sidecar());
    cleanup(&bundle);
}

#[test]
fn require_fails_on_missing_sidecar_and_loads_present_one() {
    let unsigned = write_bundle(
        "ordinaldb-sign-load-require-missing.odb",
        64,
        SignPolicy::Disabled,
    );
    let err = unwrap_err(
        open_with(&unsigned, SignLoadPolicy::Require),
        "Require must reject a bundle without a sidecar",
    );
    assert!(matches!(err, DenseError::MissingSignSidecar), "{err}");
    cleanup(&unsigned);

    let signed = write_bundle(
        "ordinaldb-sign-load-require-present.odb",
        64,
        SignPolicy::Optional,
    );
    let loaded = open_with(&signed, SignLoadPolicy::Require).expect("declared sidecar loads");
    assert!(loaded.has_sign_sidecar());
    cleanup(&signed);
}

#[test]
fn default_load_accepts_signed_sign_capable_bundle() {
    let bundle = write_bundle(
        "ordinaldb-sign-load-default-signed.odb",
        64,
        SignPolicy::Optional,
    );
    let loaded = OrdinalIndex::open_verified(
        bundle.join(MANIFEST_FILE),
        VerifyOptions::default(),
        DenseLoadOptions::default(),
    )
    .expect("signed sign-capable bundle loads by default");
    assert!(loaded.has_sign_sidecar());
    cleanup(&bundle);
}

#[test]
fn sign_incapable_bundle_loads_fine_by_default() {
    // dim=8 with bits=2 cannot carry a sidecar; RequireIfSupported does
    // not require one.
    let bundle = write_bundle("ordinaldb-sign-load-incapable.odb", 8, SignPolicy::Optional);
    let opened = OrdinalIndex::open_verified(
        bundle.join(MANIFEST_FILE),
        VerifyOptions::default(),
        DenseLoadOptions::default(),
    )
    .expect("sign-incapable bundle loads with the default policy");
    assert!(!opened.has_sign_sidecar());

    let loaded = OrdinalIndex::load(&bundle).expect("convenience load matches");
    assert!(!loaded.has_sign_sidecar());
    cleanup(&bundle);
}

#[test]
fn convenience_load_matches_open_verified_defaults() {
    // Sign-capable but unsigned: both entry points reject it.
    let unsigned = write_bundle(
        "ordinaldb-sign-load-convenience-unsigned.odb",
        64,
        SignPolicy::Disabled,
    );
    let verified_err = unwrap_err(
        OrdinalIndex::open_verified(
            unsigned.join(MANIFEST_FILE),
            VerifyOptions::default(),
            DenseLoadOptions::default(),
        ),
        "open_verified default rejects the missing sidecar",
    );
    assert!(
        matches!(verified_err, DenseError::MissingSignSidecar),
        "{verified_err}"
    );
    let load_err = match OrdinalIndex::load(&unsigned) {
        Ok(_) => panic!("load() must inherit the default policy"),
        Err(err) => err,
    };
    assert_eq!(load_err.kind(), std::io::ErrorKind::InvalidData);
    assert!(load_err.to_string().contains("sign sidecar"), "{load_err}");
    cleanup(&unsigned);

    // Signed: both entry points load it, sidecar intact.
    let signed = write_bundle(
        "ordinaldb-sign-load-convenience-signed.odb",
        64,
        SignPolicy::Optional,
    );
    let opened = OrdinalIndex::open_verified(
        signed.join(MANIFEST_FILE),
        VerifyOptions::default(),
        DenseLoadOptions::default(),
    )
    .expect("open_verified default loads the signed bundle");
    assert!(opened.has_sign_sidecar());
    let loaded = OrdinalIndex::load(&signed).expect("load() loads the signed bundle");
    assert!(loaded.has_sign_sidecar());
    cleanup(&signed);
}
