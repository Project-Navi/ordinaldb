use std::path::Path;

use ordvec_manifest::{verify_for_load, VerifiedLoadPlan, VerifyOptions};

use crate::{LtrLoadOptions, Result, TreeEnsembleReranker};

pub const DEFAULT_LTR_MODEL_AUX_NAME: &str = "ordinaldb.ltr_model";

impl TreeEnsembleReranker {
    /// Verify the bundle manifest, resolve the LTR auxiliary artifact, then
    /// immediately read and validate the model artifact.
    ///
    /// This uses `ordvec-manifest` as the path/SHA/size authority. Like the
    /// manifest verified plan itself, this is a controlled-storage boundary; it
    /// does not pin bytes against a hostile actor mutating files after
    /// verification.
    pub fn load_verified_sidecar(
        manifest_path: impl AsRef<Path>,
        aux_name: &str,
        verify_options: VerifyOptions,
        load_options: LtrLoadOptions,
    ) -> Result<Self> {
        let plan = verify_for_load(manifest_path, verify_options)?;
        Self::load_from_verified_plan_unchecked_freshness(&plan, aux_name, load_options)
    }

    /// Convenience for callers that already verified a bundle.
    ///
    /// `VerifiedLoadPlan` is a snapshot, not a byte pin. Load immediately from
    /// controlled storage, or call [`Self::load_verified_sidecar`] to re-verify
    /// before reading files another actor could mutate.
    pub fn load_from_verified_plan_unchecked_freshness(
        plan: &VerifiedLoadPlan,
        aux_name: &str,
        load_options: LtrLoadOptions,
    ) -> Result<Self> {
        let path = plan.require_auxiliary(aux_name)?;
        Self::load_unverified(path, load_options)
    }
}
