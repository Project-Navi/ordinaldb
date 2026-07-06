use ordinaldb::manifest::CreateManifestOptions;
use ordinaldb::{
    rankquant_compatible, rankquant_required_multiple, sign_compatible, sign_required_multiple,
    AddError, BuildOptions, ConstructError, DenseError, IdMapIndex, OrdinalIndex,
    OrdinalIndexBuilder, SignPolicy,
};
use ordvec::RankQuant;
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

fn with_sign(sign: SignPolicy) -> BuildOptions {
    BuildOptions { sign }
}

#[test]
fn default_build_options_are_sign_optional() {
    assert_eq!(
        BuildOptions::default(),
        BuildOptions {
            sign: SignPolicy::Optional
        }
    );
}

#[test]
fn bits_four_default_construction_works() {
    let idx = OrdinalIndex::new(16, 4).expect("bits=4 default construction");
    assert!(!idx.has_sign_sidecar());
}

#[test]
fn optional_builds_sidecar_when_dim_supports_it() {
    let mut idx = OrdinalIndex::new_with_build_options(64, 2, with_sign(SignPolicy::Optional))
        .expect("sign-capable construction");
    assert!(idx.has_sign_sidecar());

    idx.add_2d(&vectors(3, 64), 64).expect("add batch");
    let bundle = temp_path("ordinaldb-sign-policy-optional.odb");
    cleanup(&bundle);
    let report = idx
        .write_verified_bundle(&bundle, CreateManifestOptions::default(), Vec::new())
        .expect("write bundle");
    assert!(report.has_sign);
    cleanup(&bundle);
}

#[test]
fn optional_skips_sidecar_without_error_when_dim_unsupported() {
    let idx = OrdinalIndex::new_with_build_options(8, 2, with_sign(SignPolicy::Optional))
        .expect("dim=8 constructs without a sidecar");
    assert!(!idx.has_sign_sidecar());
}

#[test]
fn required_incompatible_dim_fails_at_construction() {
    let Err(err) = OrdinalIndex::new_with_build_options(8, 2, with_sign(SignPolicy::Required))
    else {
        panic!("dim=8 cannot carry a sidecar");
    };
    assert_eq!(
        err,
        ConstructError::SignSidecarUnsupported {
            dim: 8,
            bits: 2,
            required_multiple: Some(64),
        }
    );
}

#[test]
fn required_incompatible_bits_fails_at_construction() {
    let Err(err) = OrdinalIndex::new_with_build_options(16, 4, with_sign(SignPolicy::Required))
    else {
        panic!("bits=4 never carries a sidecar");
    };
    assert_eq!(
        err,
        ConstructError::SignSidecarUnsupported {
            dim: 16,
            bits: 4,
            required_multiple: None,
        }
    );
}

#[test]
fn required_compatible_builds_sidecar() {
    let idx = OrdinalIndex::new_with_build_options(64, 2, with_sign(SignPolicy::Required))
        .expect("sign-capable construction");
    assert!(idx.has_sign_sidecar());
}

#[test]
fn required_lazy_commit_rejects_incompatible_dim_and_stays_lazy() {
    let mut idx = OrdinalIndex::new_lazy_with_build_options(2, with_sign(SignPolicy::Required))
        .expect("lazy construction");

    let err = idx
        .add_2d(&vectors(2, 8), 8)
        .expect_err("dim=8 commit cannot honor Required");
    assert_eq!(
        err,
        AddError::SignSidecarUnsupported {
            dim: 8,
            bits: 2,
            required_multiple: Some(64),
        }
    );
    assert_eq!(idx.dim_opt(), None);
    assert_eq!(idx.len(), 0);

    idx.add_2d(&vectors(2, 64), 64)
        .expect("compatible dim commits");
    assert_eq!(idx.dim_opt(), Some(64));
    assert!(idx.has_sign_sidecar());
}

#[test]
fn disabled_never_builds_sidecar_even_when_compatible() {
    let mut idx = OrdinalIndex::new_with_build_options(64, 2, with_sign(SignPolicy::Disabled))
        .expect("construction");
    assert!(!idx.has_sign_sidecar());

    idx.add_2d(&vectors(3, 64), 64).expect("add batch");
    let bundle = temp_path("ordinaldb-sign-policy-disabled.odb");
    cleanup(&bundle);
    let report = idx
        .write_verified_bundle(&bundle, CreateManifestOptions::default(), Vec::new())
        .expect("write bundle");
    assert!(!report.has_sign);
    cleanup(&bundle);
}

#[test]
fn disabled_lazy_commit_never_builds_sidecar() {
    let mut idx = OrdinalIndex::new_lazy_with_build_options(2, with_sign(SignPolicy::Disabled))
        .expect("lazy construction");
    idx.add_2d(&vectors(2, 64), 64).expect("commit dim=64");
    assert!(!idx.has_sign_sidecar());
}

#[test]
fn id_map_and_builder_carry_the_policy() {
    let idx = IdMapIndex::new_with_build_options(64, 2, with_sign(SignPolicy::Disabled))
        .expect("construction");
    assert!(!idx.has_sign_sidecar());

    let Err(err) = IdMapIndex::new_with_build_options(8, 2, with_sign(SignPolicy::Required)) else {
        panic!("dim=8 cannot honor Required");
    };
    assert!(matches!(err, ConstructError::SignSidecarUnsupported { .. }));

    let mut lazy = IdMapIndex::new_lazy_with_build_options(2, with_sign(SignPolicy::Required))
        .expect("lazy construction");
    let err = lazy
        .add_with_ids_2d(&vectors(1, 8), 8, &[7])
        .expect_err("dim=8 commit cannot honor Required");
    assert_eq!(
        err,
        AddError::SignSidecarUnsupported {
            dim: 8,
            bits: 2,
            required_multiple: Some(64),
        }
    );
    assert_eq!(lazy.len(), 0);
    assert_eq!(lazy.dim_opt(), None);

    let Err(err) = OrdinalIndexBuilder::new(8, 2, with_sign(SignPolicy::Required)) else {
        panic!("builder surfaces the construct error");
    };
    assert!(matches!(
        err,
        DenseError::Construct(ConstructError::SignSidecarUnsupported { .. })
    ));
}

#[test]
fn preflight_helpers_report_required_multiples() {
    assert_eq!(rankquant_required_multiple(1), Some(8));
    assert_eq!(rankquant_required_multiple(2), Some(4));
    assert_eq!(rankquant_required_multiple(4), Some(16));
    assert_eq!(rankquant_required_multiple(3), None);
    assert_eq!(rankquant_required_multiple(8), None);

    assert_eq!(sign_required_multiple(2), Some(64));
    assert_eq!(sign_required_multiple(1), None);
    assert_eq!(sign_required_multiple(4), None);
}

#[test]
fn preflight_compatibility_matches_construction_outcomes() {
    for bits in [1u8, 2, 4] {
        let multiple = rankquant_required_multiple(bits).unwrap();
        for dim in 0..=257usize {
            let compatible = rankquant_compatible(dim, bits);
            assert_eq!(
                compatible,
                OrdinalIndex::new(dim, bits).is_ok(),
                "dim={dim} bits={bits}"
            );
            assert_eq!(
                compatible,
                dim >= 2 && dim.is_multiple_of(multiple),
                "dim={dim} bits={bits}"
            );
        }
    }
    assert!(!rankquant_compatible(1 << 17, 2), "dim beyond u16::MAX");
    assert!(!rankquant_compatible(64, 3), "unsupported bits");
}

#[test]
fn preflight_compatibility_agrees_with_ordvec_ground_truth() {
    for bits in [1u8, 2, 4] {
        for dim in 0..=257usize {
            assert_eq!(
                rankquant_compatible(dim, bits),
                RankQuant::validate_params(dim, bits).is_ok(),
                "dim={dim} bits={bits}"
            );
        }
    }
}

#[test]
fn sign_compatible_matches_sidecar_construction() {
    for bits in [1u8, 2, 4] {
        for dim in [8usize, 16, 32, 64, 96, 128, 192, 256] {
            if !rankquant_compatible(dim, bits) {
                assert!(!sign_compatible(dim, bits), "dim={dim} bits={bits}");
                continue;
            }
            let idx =
                OrdinalIndex::new_with_build_options(dim, bits, with_sign(SignPolicy::Optional))
                    .expect("compatible construction");
            assert_eq!(
                idx.has_sign_sidecar(),
                sign_compatible(dim, bits),
                "dim={dim} bits={bits}"
            );
        }
    }
    assert!(sign_compatible(64, 2));
    assert!(!sign_compatible(8, 2));
    assert!(!sign_compatible(64, 1));
    assert!(!sign_compatible(64, 4));
}
