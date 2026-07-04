"""
Operational workflow for using RuuPhenome in a real NMR metabolomics laboratory.

The software currently supports research use. This module makes the surrounding
laboratory process explicit: who performs each step, which records are required,
where quality gates occur, and which parts map to the existing API.
"""

from __future__ import annotations

from typing import Dict, List, Optional


WORKFLOW_VERSION = "2026.1"


def _stage(
    stage_id: str,
    order: int,
    name: str,
    owner: str,
    objective: str,
    inputs: List[str],
    actions: List[str],
    records: List[str],
    gate: List[str],
    failure_action: str,
    app_mapping: str,
) -> Dict:
    return {
        "id": stage_id,
        "order": order,
        "name": name,
        "owner": owner,
        "objective": objective,
        "inputs": inputs,
        "actions": actions,
        "required_records": records,
        "release_gate": gate,
        "failure_action": failure_action,
        "ruuphenome_mapping": app_mapping,
    }


STAGES = [
    _stage(
        "study_setup",
        1,
        "Study setup and method lock",
        "Principal investigator + laboratory director",
        "Define the intended use before samples enter the laboratory.",
        ["Approved protocol", "Target sample matrix", "Outcome definition"],
        [
            "Assign study, batch and SOP identifiers.",
            "Predefine inclusion, exclusion and sample-rejection rules.",
            "Lock acquisition, processing, QC and statistical-analysis versions.",
            "Create a randomized run order stratified by important study groups.",
        ],
        [
            "Protocol/SOP versions",
            "Sample-size rationale",
            "Randomization seed and run order",
            "Model and reference-library versions",
        ],
        [
            "Ethics/consent requirements satisfied where applicable",
            "Intended use is explicitly research, validation or clinical",
            "Acceptance criteria approved before acquisition",
        ],
        "Do not receive study samples until the protocol and acceptance rules are approved.",
        "Configuration is external today; store identifiers with every API result.",
    ),
    _stage(
        "accession",
        2,
        "Sample accession and chain of custody",
        "Accessioning technician",
        "Create an unambiguous, auditable identity for every specimen.",
        ["Specimen", "Submission manifest", "Collection metadata"],
        [
            "Assign a barcode and pseudonymous sample identifier.",
            "Record collection, receipt, processing and freezer timestamps.",
            "Record matrix, tube/additive, volume, storage temperature and freeze-thaw count.",
            "Reconcile the physical tube against the electronic manifest.",
        ],
        [
            "Barcode and sample ID",
            "Chain-of-custody events",
            "Collector and receiver identities",
            "Collection/transport/storage metadata",
        ],
        [
            "Tube and manifest identities agree",
            "Required timestamps and matrix metadata are present",
            "No unresolved custody break",
        ],
        "Quarantine the specimen; never repair identity discrepancies by assumption.",
        "Requires LIMS integration; do not use filenames as the primary sample identity.",
    ),
    _stage(
        "preanalytical_qc",
        3,
        "Pre-analytical acceptance",
        "Laboratory technician",
        "Prevent unsuitable specimens from entering the analytical batch.",
        ["Accessioned specimen", "Matrix-specific acceptance SOP"],
        [
            "Inspect volume, container integrity and visible contamination.",
            "Record hemolysis, lipemia or icterus where relevant to the matrix.",
            "Check transport temperature, processing delay and freeze-thaw limits.",
            "Accept, reject or quarantine with a reason code.",
        ],
        [
            "Acceptance decision",
            "Deviation/rejection reason",
            "Condition observations",
            "Author and timestamp",
        ],
        [
            "Minimum volume available",
            "Matrix and collection tube are permitted",
            "Time/temperature and freeze-thaw limits are met",
        ],
        "Reject or quarantine according to the locked SOP; notify the study owner.",
        "Manual/LIMS gate before `/process-fid`.",
    ),
    _stage(
        "sample_preparation",
        4,
        "Standardized sample preparation",
        "NMR laboratory technician",
        "Prepare study samples and controls using one locked procedure.",
        ["Accepted specimen", "Validated buffer", "Internal standard", "D2O"],
        [
            "Thaw, mix and centrifuge using controlled time and temperature.",
            "Add fixed volumes of sample, buffer, D2O and internal standard.",
            "Prepare a pooled study QC from representative aliquots.",
            "Prepare process blank, solvent blank and reference/control material.",
            "Transfer to labelled NMR tubes and record preparation order.",
        ],
        [
            "Reagent lots and expiry dates",
            "Pipette/instrument IDs",
            "Preparation volumes and timestamps",
            "Operator and deviations",
        ],
        [
            "Internal standard and pH target are within the SOP range",
            "Required blank, pooled QC and reference materials exist",
            "Tube barcode matches the prepared sample",
        ],
        "Reprepare before acquisition when possible; otherwise quarantine and document.",
        "Internal-standard detection is checked by Domain 1 processing.",
    ),
    _stage(
        "batch_design",
        5,
        "Analytical batch assembly",
        "NMR scientist",
        "Make drift, contamination and batch effects measurable.",
        ["Randomized study samples", "Pooled QC", "Blanks", "Reference material"],
        [
            "Begin with instrument-suitability and conditioning injections/acquisitions.",
            "Place pooled QC at the beginning, end and at the SOP-defined interval.",
            "Place blanks after high-concentration or carry-over-risk samples.",
            "Preserve the locked randomization unless a deviation is recorded.",
        ],
        [
            "Final plate/rack map",
            "Acquisition queue",
            "QC and blank positions",
            "Run-order deviations",
        ],
        [
            "Every tube has a unique queue position",
            "QC frequency is sufficient to estimate drift",
            "Study groups are not confounded with run order",
        ],
        "Rebuild the queue before acquisition if controls or randomization are inadequate.",
        "Batch metadata is not yet persisted by the app.",
    ),
    _stage(
        "instrument_suitability",
        6,
        "Instrument readiness and system suitability",
        "NMR operator",
        "Demonstrate that the spectrometer is fit for the locked method.",
        ["Suitability sample", "Locked acquisition method", "Maintenance status"],
        [
            "Confirm probe, temperature, tune/match, lock, shim and receiver settings.",
            "Acquire the system-suitability/reference sample.",
            "Check line shape/width, chemical-shift reference, water suppression and sensitivity.",
            "Record software, pulse-program and parameter versions.",
        ],
        [
            "Instrument and probe IDs",
            "Pulse sequence and acquisition parameters",
            "Suitability metrics",
            "Pass/fail decision and operator",
        ],
        [
            "Preventive maintenance and calibrations are current",
            "Suitability metrics meet the laboratory's validated limits",
            "No unresolved instrument alarm",
        ],
        "Stop the batch, troubleshoot, repeat suitability and document corrective action.",
        "Acquisition metadata is read from the Bruker archive by `/process-fid`.",
    ),
    _stage(
        "acquisition",
        7,
        "1D 1H NMR acquisition",
        "NMR operator",
        "Acquire comparable raw data while preserving the original evidence.",
        ["Released queue", "Suitability pass", "Prepared tubes"],
        [
            "Acquire the locked 1D 1H experiment for every sample and control.",
            "Monitor lock, shim, temperature and receiver warnings.",
            "Store the original Bruker experiment directory as read-only raw data.",
            "Calculate a checksum before transfer to analysis storage.",
        ],
        [
            "Raw FID and acquisition files",
            "Queue/run log",
            "Warnings and deviations",
            "Raw-data checksum",
        ],
        [
            "Raw FID is complete and readable",
            "Sample ID matches queue and tube",
            "Acquisition used the locked parameter set or has an approved deviation",
        ],
        "Reacquire before tube disposal when an acquisition failure is recoverable.",
        "Zip one Bruker experiment and send it to `POST /process-fid`.",
    ),
    _stage(
        "signal_processing",
        8,
        "Automated signal processing",
        "RuuPhenome service",
        "Convert raw FIDs into referenced, quality-scored spectra reproducibly.",
        ["Immutable Bruker FID", "Acquisition metadata", "Locked processing settings"],
        [
            "Validate the archive and remove the Bruker digital filter.",
            "Apply DC correction, apodization, zero fill, FFT and automatic phasing.",
            "Correct baseline, reference DSS/TSP/TMS and normalize the spectrum.",
            "Estimate noise, pick peaks and produce processing/QC metadata.",
        ],
        [
            "Processing parameters and software versions",
            "Processed spectrum",
            "Peaks and QC metrics",
            "Input checksum and output identifier",
        ],
        [
            "Processing completes without an exception",
            "No unsafe or malformed archive content",
            "All processing steps and versions are retained",
        ],
        "Quarantine the result; inspect the raw data and repeat only under a documented rule.",
        "`POST /process-fid` or `POST /process-spectrum`.",
    ),
    _stage(
        "spectrum_qc",
        9,
        "Spectrum and batch QC release",
        "NMR scientist",
        "Release only technically valid spectra for interpretation.",
        ["Processed spectrum", "Per-spectrum QC", "Pooled QC series", "Blanks"],
        [
            "Review QC score, SNR, phasing, baseline and chemical-shift referencing.",
            "Check pooled-QC precision and signal drift across run order.",
            "Check blanks for contamination or carry-over.",
            "Assign pass, fail or needs-review with reason codes.",
        ],
        [
            "Automated rule results",
            "Pooled-QC and blank metrics",
            "Manual review decision",
            "Reviewer and timestamp",
        ],
        [
            "Per-spectrum limits pass",
            "Pooled-QC precision and drift meet the validated SOP",
            "Blank contamination is absent",
        ],
        "Reprocess only for predefined technical reasons; otherwise reacquire or exclude.",
        "`POST /laboratory-workflow/evaluate-qc` provides the software-side gate.",
    ),
    _stage(
        "identification",
        10,
        "Metabolite identification and quantification",
        "Metabolomics analyst",
        "Convert released spectral evidence into reviewed metabolite results.",
        ["QC-released spectrum", "Locked reference library", "Validated standards"],
        [
            "Review complete resonance-pattern matches and ppm errors.",
            "Use self-supervised similarity as supporting evidence, not sole identification.",
            "Resolve overlaps and ambiguous assignments with standards or orthogonal evidence.",
            "Quantify only with a validated calibration/internal-standard method.",
        ],
        [
            "Assignment evidence and confidence",
            "Library/model versions",
            "Manual edits and rationale",
            "Calibration and concentration units",
        ],
        [
            "Identification confidence meets the intended-use threshold",
            "Ambiguous assignments remain labelled as ambiguous",
            "Reported concentrations have validated traceability",
        ],
        "Hold uncertain identifications for review; never convert pseudo-abundance to clinical concentration.",
        "Domain 1 assignments plus `self_supervised_matches`; current concentrations are not clinically calibrated.",
    ),
    _stage(
        "cohort_analysis",
        11,
        "Cohort QC and statistical analysis",
        "Biostatistician + metabolomics analyst",
        "Detect batch structure and estimate biological associations without leakage.",
        ["Released result matrix", "Blinded metadata", "Patient/group identifiers"],
        [
            "Use PCA for outliers, drift and batch-effect inspection.",
            "Use UMAP only as exploratory visualization, with parameters and seed recorded.",
            "Apply normalization/correction using controls and training data only.",
            "Keep patients together and feature selection inside cross-validation folds.",
            "Unblind outcomes only according to the analysis plan.",
        ],
        [
            "Analysis code and environment",
            "PCA loadings/scores and QC decisions",
            "Cross-validation splits and random seeds",
            "Model metrics, calibration and stable features",
        ],
        [
            "No unresolved batch/run-order confounding",
            "No preprocessing or feature-selection leakage",
            "Performance includes uncertainty and independent validation where required",
        ],
        "Investigate batch effects; do not remove inconvenient samples without a predefined, documented rule.",
        "Domain 2 leakage-safe and model-suite endpoints; PCA/UMAP are recommended next additions.",
    ),
    _stage(
        "review_report",
        12,
        "Independent review and report authorization",
        "Second qualified reviewer + laboratory director",
        "Ensure the report matches the evidence and intended use.",
        ["QC decisions", "Assignments/results", "Statistical output", "Deviations"],
        [
            "Perform technical and scientific review by someone other than the primary analyst.",
            "Confirm sample identity, units, flags, methods and limitations.",
            "Separate research findings from clinically validated claims.",
            "Electronically authorize the final version.",
        ],
        [
            "Reviewer comments",
            "Resolved discrepancies",
            "Authorized report",
            "Report version and signatures",
        ],
        [
            "All failed/held samples are excluded or clearly flagged",
            "Limitations and intended use are visible",
            "Two-person review is complete",
        ],
        "Return to the responsible stage; preserve the rejected report version and reason.",
        "Reporting/signature workflow requires LIMS or an electronic quality system.",
    ),
    _stage(
        "archive",
        13,
        "Release, archive and continual improvement",
        "Quality manager + data steward",
        "Make every released result reproducible and recoverable.",
        ["Authorized report", "Raw and processed data", "Audit trail"],
        [
            "Release the report only to authorized recipients.",
            "Archive raw FIDs, checksums, processed spectra, metadata, models and reports.",
            "Back up and periodically test restoration.",
            "Trend QC failures, deviations and corrective/preventive actions.",
        ],
        [
            "Release log",
            "Immutable archive manifest",
            "Backup/restore evidence",
            "Deviation and CAPA trend",
        ],
        [
            "Archive contains every input, parameter, version and decision needed to reproduce the result",
            "Retention and access policies are applied",
            "Backup restoration has been tested",
        ],
        "Do not release if the analysis cannot be traced or reproduced.",
        "Checksums/provenance exist for open data; production deployment still needs LIMS, authentication and durable audit storage.",
    ),
]


def workflow() -> Dict:
    """Return the complete machine-readable real-laboratory workflow."""
    return {
        "name": "RuuPhenome real-laboratory NMR metabolomics workflow",
        "version": WORKFLOW_VERSION,
        "intended_use": (
            "Research use and method-development workflow. It is not a clinical "
            "diagnostic procedure until the complete laboratory method, software, "
            "personnel and reporting process are validated for the intended use."
        ),
        "operating_principles": [
            "Sample identity and chain of custody precede analysis.",
            "Raw FIDs are immutable; reprocessing creates a new version.",
            "QC acceptance criteria are locked before the batch is acquired.",
            "Pooled QC, blanks and reference material travel through the batch.",
            "Machine-learning evidence supports but does not replace chemical review.",
            "A second qualified person authorizes release.",
        ],
        "standards_alignment": [
            {
                "name": "ISO 15189:2022",
                "scope": "Medical-laboratory quality and competence framework",
                "url": "https://www.iso.org/standard/76677.html",
            },
            {
                "name": "ISO 23118:2021",
                "scope": "Pre-examination processes for metabolomics in urine, serum and plasma",
                "url": "https://www.iso.org/standard/74605.html",
            },
            {
                "name": "ICH M10",
                "scope": "Fit-for-purpose bioanalytical validation and study-sample analysis principles",
                "url": "https://www.fda.gov/media/162903/download",
            },
            {
                "name": "Metabolomics Standards Initiative",
                "scope": "Minimum reporting for preparation, analysis, QC, identification and preprocessing",
                "url": "https://doi.org/10.1007/s11306-007-0082-2",
            },
            {
                "name": "MetaboLights guides",
                "scope": "Study, sample and protocol metadata structure",
                "url": "https://www.ebi.ac.uk/metabolights/editor/guides",
            },
        ],
        "stages": STAGES,
        "minimum_production_components": [
            "LIMS/barcode integration and role-based access",
            "Electronic signatures and append-only audit trail",
            "Validated matrix-specific sample-preparation and acquisition SOPs",
            "External reference materials and proficiency testing",
            "Validated quantitative calibration where concentrations are reported",
            "Independent clinical/performance validation for any diagnostic claim",
            "Backup, disaster recovery, cybersecurity and retention policy",
        ],
    }


def _rule(
    rule_id: str,
    label: str,
    status: str,
    observed,
    acceptance: str,
    action: str,
) -> Dict:
    return {
        "id": rule_id,
        "label": label,
        "status": status,
        "observed": observed,
        "acceptance": acceptance,
        "action": action,
    }


def evaluate_qc(
    *,
    qc_score: Optional[float] = None,
    max_snr: Optional[float] = None,
    negative_area_fraction: Optional[float] = None,
    reference_method: Optional[str] = None,
    pooled_qc_cv_percent: Optional[float] = None,
    drift_percent: Optional[float] = None,
    blank_contamination: Optional[bool] = None,
    instrument_suitability_passed: Optional[bool] = None,
    sample_identity_verified: Optional[bool] = None,
) -> Dict:
    """
    Apply conservative default research-use QC gates.

    These defaults are deliberately visible and configurable in a future LIMS.
    A laboratory must replace them with matrix- and method-validated limits
    before clinical or regulated use.
    """
    rules = []

    def numeric_rule(
        rule_id: str,
        label: str,
        observed: Optional[float],
        operator: str,
        limit: float,
        acceptance: str,
        action: str,
    ) -> None:
        if observed is None:
            status = "needs_review"
        elif operator == ">=":
            status = "pass" if observed >= limit else "fail"
        else:
            status = "pass" if observed <= limit else "fail"
        rules.append(_rule(rule_id, label, status, observed, acceptance, action))

    numeric_rule(
        "spectrum_qc_score",
        "Automated spectrum QC score",
        qc_score,
        ">=",
        75.0,
        "score >= 75",
        "Review processing and raw FID; reacquire if a technical failure remains.",
    )
    numeric_rule(
        "maximum_snr",
        "Maximum signal-to-noise ratio",
        max_snr,
        ">=",
        20.0,
        "max SNR >= 20",
        "Inspect sample concentration, receiver settings and acquisition integrity.",
    )
    numeric_rule(
        "negative_area",
        "Negative spectral area",
        negative_area_fraction,
        "<=",
        0.20,
        "negative area fraction <= 0.20",
        "Review phasing and baseline correction.",
    )

    reference = (reference_method or "").strip().casefold()
    if not reference_method:
        reference_status = "needs_review"
    elif reference == "internal standard":
        reference_status = "pass"
    else:
        reference_status = "needs_review"
    rules.append(
        _rule(
            "chemical_shift_reference",
            "Chemical-shift reference",
            reference_status,
            reference_method,
            "internal-standard reference confirmed",
            "Confirm DSS/TSP/TMS manually or apply the laboratory's approved reference rule.",
        )
    )

    numeric_rule(
        "pooled_qc_precision",
        "Pooled-QC precision",
        pooled_qc_cv_percent,
        "<=",
        20.0,
        "pooled-QC CV <= 20%",
        "Investigate preparation, instrument stability and unsuitable features.",
    )
    numeric_rule(
        "batch_drift",
        "Batch signal drift",
        drift_percent,
        "<=",
        20.0,
        "absolute drift <= 20%",
        "Investigate run order and instrument stability before correction or release.",
    )

    for rule_id, label, observed, action in (
        (
            "blank_contamination",
            "Blank contamination/carry-over",
            blank_contamination,
            "Locate the contamination source and repeat affected blanks/samples.",
        ),
        (
            "instrument_suitability",
            "Instrument suitability",
            instrument_suitability_passed,
            "Stop the batch and restore instrument suitability.",
        ),
        (
            "sample_identity",
            "Sample identity verification",
            sample_identity_verified,
            "Quarantine the specimen and resolve chain of custody.",
        ),
    ):
        if observed is None:
            status = "needs_review"
        elif rule_id == "blank_contamination":
            status = "fail" if observed else "pass"
        else:
            status = "pass" if observed else "fail"
        acceptance = (
            "no contamination detected"
            if rule_id == "blank_contamination"
            else "confirmed"
        )
        rules.append(_rule(rule_id, label, status, observed, acceptance, action))

    statuses = {item["status"] for item in rules}
    if "fail" in statuses:
        decision = "fail"
        release = False
    elif "needs_review" in statuses:
        decision = "needs_review"
        release = False
    else:
        decision = "pass"
        release = True

    return {
        "decision": decision,
        "release_allowed": release,
        "rules": rules,
        "policy": {
            "version": WORKFLOW_VERSION,
            "status": "default research-use limits",
            "warning": (
                "Replace these defaults with validated, matrix-specific limits "
                "before regulated or clinical use."
            ),
        },
    }
