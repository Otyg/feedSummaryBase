# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""Deterministic primitives for incremental threat-landscape analysis."""

from feedsummary_core.long_term.clustering import (
    AssignmentAction,
    AssignmentDecision,
    ClusteringSettings,
    VectorValidationError,
    add_cluster_member,
    assign_article,
    cluster_status_at,
    cosine_similarity,
    create_cluster,
    stable_cluster_id,
)
from feedsummary_core.long_term.lease import (
    LeaseGuard,
    LeaseLostError,
    LongTermLeaseHeartbeat,
)
from feedsummary_core.long_term.map_analysis import (
    MapSettings,
    MapUpdateResult,
    PromptBudgetError,
    SnapshotRevisionConflict,
    cluster_needs_map_update,
    update_cluster_map_snapshot,
)
from feedsummary_core.long_term.metrics import (
    LandscapeMetricSettings,
    build_landscape_metrics,
    compute_landscape_metrics,
)
from feedsummary_core.long_term.models import (
    ACTIVE_CLUSTER_STATUSES,
    ClusterStatus,
    EmbeddingSignature,
    ThreatCluster,
)
from feedsummary_core.long_term.processor import (
    ArticleAssignment,
    ArticleQualityError,
    ConcurrentAssignmentError,
    IncrementalBatchResult,
    IncrementalSettings,
    LeaseUnavailableError,
    run_incremental_clustering,
)
from feedsummary_core.long_term.reduce_analysis import (
    ReduceBudgetError,
    ReduceResult,
    ReduceSettings,
    ReportMirrorError,
    mirror_landscape_report,
    run_landscape_reduce,
    select_report_snapshots,
    validate_landscape_segment,
)
from feedsummary_core.long_term.reporting import (
    ReportValidationError,
    build_landscape_report_document,
    build_landscape_summary_document,
    parse_landscape_report_json,
    render_landscape_markdown,
    render_landscape_report_messages,
    validate_landscape_report,
)
from feedsummary_core.long_term.snapshot_validation import (
    SnapshotValidationError,
    parse_snapshot_json,
    validate_cluster_snapshot,
)

__all__ = [
    "ACTIVE_CLUSTER_STATUSES",
    "ArticleAssignment",
    "ArticleQualityError",
    "AssignmentAction",
    "AssignmentDecision",
    "ClusterStatus",
    "ClusteringSettings",
    "ConcurrentAssignmentError",
    "EmbeddingSignature",
    "IncrementalBatchResult",
    "IncrementalSettings",
    "LandscapeMetricSettings",
    "LeaseGuard",
    "LeaseLostError",
    "LeaseUnavailableError",
    "LongTermLeaseHeartbeat",
    "MapSettings",
    "MapUpdateResult",
    "PromptBudgetError",
    "ReduceBudgetError",
    "ReduceResult",
    "ReduceSettings",
    "ReportMirrorError",
    "ReportValidationError",
    "SnapshotRevisionConflict",
    "SnapshotValidationError",
    "ThreatCluster",
    "VectorValidationError",
    "add_cluster_member",
    "assign_article",
    "build_landscape_metrics",
    "build_landscape_report_document",
    "build_landscape_summary_document",
    "cluster_needs_map_update",
    "cluster_status_at",
    "compute_landscape_metrics",
    "cosine_similarity",
    "create_cluster",
    "mirror_landscape_report",
    "parse_landscape_report_json",
    "parse_snapshot_json",
    "render_landscape_markdown",
    "render_landscape_report_messages",
    "run_incremental_clustering",
    "run_landscape_reduce",
    "select_report_snapshots",
    "stable_cluster_id",
    "update_cluster_map_snapshot",
    "validate_cluster_snapshot",
    "validate_landscape_report",
    "validate_landscape_segment",
]
