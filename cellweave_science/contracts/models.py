"""
CellWeave Cross-Language Contracts — Generated Pydantic v2 Models.

AUTOMATICALLY GENERATED FROM CANONICAL JSON SCHEMAS & OPENAPI 3.1.
DO NOT EDIT MANUALLY.

Single source of truth: /schemas/ and /openapi/
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


# --- Domain State Enums ---

class RunState(str, Enum):
    DRAFT = "DRAFT"
    UPLOADING = "UPLOADING"
    VALIDATING = "VALIDATING"
    QC_RUNNING = "QC_RUNNING"
    QC_REVIEW_REQUIRED = "QC_REVIEW_REQUIRED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED_QUOTA = "PAUSED_QUOTA"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class WorkflowStepState(str, Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    RETRYING = "RETRYING"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


class UploadState(str, Enum):
    CREATED = "CREATED"
    UPLOADING = "UPLOADING"
    UPLOADED = "UPLOADED"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    IMPORTED = "IMPORTED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    EXPIRED = "EXPIRED"
    QUARANTINED = "QUARANTINED"


class AccountState(str, Enum):
    ACTIVE = "ACTIVE"
    PAST_DUE = "PAST_DUE"
    SUSPENDED = "SUSPENDED"
    CLOSING = "CLOSING"
    DELETED = "DELETED"


class ResourceClass(str, Enum):
    CPU_SMALL = "CPU_SMALL"
    CPU_MEMORY = "CPU_MEMORY"
    CPU_HIGH_MEMORY = "CPU_HIGH_MEMORY"
    GPU_SMALL = "GPU_SMALL"
    GPU_LARGE = "GPU_LARGE"


class ArtifactType(str, Enum):
    SOURCE_DATASET = "SOURCE_DATASET"
    CANONICAL_DATASET = "CANONICAL_DATASET"
    QC_REPORT = "QC_REPORT"
    NORMALIZED_DATASET = "NORMALIZED_DATASET"
    EMBEDDING = "EMBEDDING"
    CLUSTERS = "CLUSTERS"
    ANNOTATIONS = "ANNOTATIONS"
    MARKER_TABLE = "MARKER_TABLE"
    DE_TABLE = "DE_TABLE"
    PATHWAY_TABLE = "PATHWAY_TABLE"
    FIGURE = "FIGURE"
    REPORT = "REPORT"
    RUN_MANIFEST = "RUN_MANIFEST"
    EXPORT_BUNDLE = "EXPORT_BUNDLE"


class ScientificJobType(str, Enum):
    VALIDATE_DATASET = "validate_dataset"
    RUN_QC = "run_qc"
    FILTER_CELLS = "filter_cells"
    NORMALIZE = "normalize"
    SELECT_HVG = "select_hvg"
    RUN_PCA = "run_pca"
    BUILD_NEIGHBORS = "build_neighbors"
    RUN_UMAP = "run_umap"
    CLUSTER = "cluster"
    ANNOTATE = "annotate"
    FIND_MARKERS = "find_markers"
    RUN_DIFFERENTIAL_EXPRESSION = "run_differential_expression"
    RUN_PATHWAY_ANALYSIS = "run_pathway_analysis"
    GENERATE_EXPORTS = "generate_exports"


# --- Domain Error Model ---

class DomainError(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="User-facing safe description")
    technicalDetail: str = Field(..., description="Diagnostic context excluding PHI and raw data")
    recommendedAction: str = Field(..., description="Resolution guidance")
    correlationId: str = Field(..., description="Unique request or run trace correlation ID")
    retryable: bool = Field(False, description="Whether retry is permitted without change")
    details: Optional[Dict[str, Any]] = Field(None, description="Structured error details")


# --- Compute Provider Models ---

class ComputeJobSubmitRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    jobId: str
    jobType: str
    resourceClass: ResourceClass
    payloadUri: str
    maxRuntimeSeconds: int = Field(..., ge=1)
    correlationId: str
    tags: Optional[Dict[str, str]] = None


class ComputeJobStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    jobId: str
    providerJobId: str
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]
    startedAt: str
    completedAt: Optional[str] = None
    exitCode: Optional[int] = None
    errorMessage: Optional[str] = None


class ComputeJobCancelRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    jobId: str
    providerJobId: str
    reason: str


class ComputeJobMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cpuSeconds: float = Field(..., ge=0)
    gpuSeconds: float = Field(0.0, ge=0)
    memoryMegabyteSeconds: float = Field(..., ge=0)
    peakMemoryBytes: Optional[int] = Field(None, ge=0)


class ComputeJobResultResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    jobId: str
    providerJobId: str
    status: Literal["COMPLETED", "FAILED", "CANCELLED"]
    metrics: ComputeJobMetrics
    resultUri: Optional[str] = None
    error: Optional[DomainError] = None


# --- Scientific Job Payload Models ---

class ScopedPrimaryDatasetInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    downloadUrl: str
    sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    format: Optional[str] = None


class ScopedAuxiliaryInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    downloadUrl: str
    sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")


class ScopedOutputTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    uploadUrl: str
    expectedMimeType: str


class ScientificJobInputs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    primaryDataset: ScopedPrimaryDatasetInput
    auxiliaryInputs: Optional[List[ScopedAuxiliaryInput]] = None


class ScientificJobOutputs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    targets: List[ScopedOutputTarget]


class ScientificJobManifestFragment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    containerDigest: str
    randomSeeds: Dict[str, int]
    softwareVersions: Optional[Dict[str, str]] = None


class ScientificJobPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schemaVersion: str = Field(..., pattern=r"^1.[0-9]+.[0-9]+$")
    pipelineVersion: str
    jobId: str
    runId: str
    stepId: str
    jobType: ScientificJobType
    inputs: ScientificJobInputs
    outputs: ScientificJobOutputs
    parameters: Dict[str, Any]
    manifestFragment: ScientificJobManifestFragment
    resourceClass: Optional[ResourceClass] = None


# --- Manifest Models ---

class PipelineRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str


class ContainerRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    digest: str = Field(..., pattern=r"^sha256:[a-f0-9]{64}$")
    image: Optional[str] = None


class InputDatasetRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    datasetVersionId: str
    sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    sourceFormat: Optional[str] = None


class RunManifestStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stepId: str
    name: str
    status: WorkflowStepState
    startedAt: str
    completedAt: str
    version: Optional[str] = None
    computeUnits: Optional[float] = None
    resourceClass: Optional[ResourceClass] = None
    outputArtifactIds: Optional[List[str]] = None


class RunManifestArtifactSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifactId: str
    type: str
    sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    objectKey: str
    sizeBytes: Optional[int] = None


class ScientificWarning(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    message: str
    step: str
    details: Optional[Dict[str, Any]] = None


class ManifestTimestamps(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    startedAt: str
    completedAt: str


class RunManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schemaVersion: str = Field(..., pattern=r"^1.[0-9]+.[0-9]+$")
    runId: str
    pipeline: PipelineRef
    container: ContainerRef
    input: InputDatasetRef
    organism: Literal["human", "mouse"]
    parameters: Dict[str, Any]
    references: Dict[str, Any]
    software: Dict[str, str]
    randomSeeds: Dict[str, int]
    steps: List[RunManifestStep]
    artifacts: List[RunManifestArtifactSummary]
    timestamps: ManifestTimestamps
    warnings: List[ScientificWarning] = Field(default_factory=list)


class ArtifactLineageRelation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parentArtifactId: str
    relationship: Literal["DERIVED_FROM", "SUMMARIZES", "BUNDLES", "TRANSFORMS"]


class ArtifactStorageRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["R2", "S3_COMPATIBLE"]
    bucket: str
    objectKey: str


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schemaVersion: str = Field(..., pattern=r"^1.[0-9]+.[0-9]+$")
    artifactId: str
    runId: str
    type: ArtifactType
    storage: ArtifactStorageRef
    sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    sizeBytes: int = Field(..., ge=0)
    mimeType: str
    producingStepId: str
    lineage: List[ArtifactLineageRelation]
    createdAt: str
    metadata: Optional[Dict[str, Any]] = None


# --- Event Models ---

class UsageMeasurements(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cpuSeconds: float = Field(..., ge=0)
    gpuSeconds: float = Field(0.0, ge=0)
    memoryClass: Optional[ResourceClass] = None
    bytesProcessed: Optional[int] = Field(None, ge=0)
    storageByteHours: Optional[float] = Field(None, ge=0)
    egressBytes: Optional[int] = Field(None, ge=0)


class UsageEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schemaVersion: str = Field(..., pattern=r"^1.[0-9]+.[0-9]+$")
    idempotencyKey: str = Field(..., pattern=r"^.+:.+:[0-9]+$")
    organizationId: str
    projectId: str
    runId: str
    stepId: str
    attempt: int = Field(..., ge=1)
    measurements: UsageMeasurements
    timestamp: str


class AuditActor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    actorId: str
    actorType: Literal["USER", "SYSTEM", "SUPPORT_STAFF_BREAKGLASS"]
    ipAddress: Optional[str] = None
    userAgent: Optional[str] = None


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schemaVersion: str = Field(..., pattern=r"^1.[0-9]+.[0-9]+$")
    eventId: str
    organizationId: str
    actor: AuditActor
    action: str
    resourceType: str
    resourceId: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class ServiceEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schemaVersion: str = Field(..., pattern=r"^1.[0-9]+.[0-9]+$")
    eventId: str
    eventType: str
    source: str
    correlationId: str
    data: Dict[str, Any]
    timestamp: str
    organizationId: Optional[str] = None


class PolarWebhookData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    status: Optional[str] = None
    organizationId: Optional[str] = None
    customerId: Optional[str] = None
    subscriptionId: Optional[str] = None
    currentPeriodEnd: Optional[str] = None
    cancelAtPeriodEnd: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class PolarWebhookPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schemaVersion: str = Field(..., pattern=r"^1.[0-9]+.[0-9]+$")
    eventId: str
    type: str
    createdAt: str
    data: PolarWebhookData


# --- OpenAPI Phase 0 API Models ---

class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    schemaVersion: str
    timestamp: str
