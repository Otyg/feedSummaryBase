import copy
import unittest

from feedsummary_core.long_term import (
    LandscapeMetricSettings,
    build_landscape_metrics,
    compute_landscape_metrics,
)

DAY = 86400


def cluster(
    cluster_id,
    *,
    first_seen,
    last_seen,
    status="active",
    membership_revision=1,
    summarized_revision=1,
    member_count=None,
    last_reopened_at=None,
):
    return {
        "id": cluster_id,
        "profile_id": "profile",
        "status": status,
        "first_seen_ts": first_seen,
        "last_seen_ts": last_seen,
        "membership_revision": membership_revision,
        "summarized_revision": summarized_revision,
        "member_count": member_count or membership_revision,
        "last_reopened_at": last_reopened_at,
    }


def membership(cluster_id, article_id, article_ts, source=None):
    row = {
        "profile_id": "profile",
        "cluster_id": cluster_id,
        "article_id": article_id,
        "article_ts": article_ts,
    }
    if source is not None:
        row["evidence"] = {"source": source, "published_ts": article_ts}
    return row


class FakeStore:
    def __init__(self, clusters, memberships, articles=()):
        self.clusters = list(clusters)
        self.memberships = list(memberships)
        self.articles = list(articles)

    def list_threat_clusters(self, profile_id, **kwargs):
        return [row for row in self.clusters if row["profile_id"] == profile_id]

    def list_cluster_memberships(self, cluster_id, *, limit=10000):
        return [
            row for row in self.memberships if row["cluster_id"] == cluster_id
        ][:limit]

    def get_articles_by_ids(self, article_ids):
        wanted = set(article_ids)
        return [row for row in self.articles if row["id"] in wanted]


class LongTermMetricsTests(unittest.TestCase):
    def setUp(self):
        self.end = 200 * DAY
        self.clusters = [
            cluster(
                "c1",
                first_seen=self.end - 5 * DAY,
                last_seen=self.end - DAY,
                membership_revision=2,
                summarized_revision=2,
            ),
            cluster(
                "c2",
                first_seen=self.end - 20 * DAY,
                last_seen=self.end - 7 * DAY,
                status="dormant",
                last_reopened_at=self.end - 8 * DAY,
            ),
            cluster(
                "c3",
                first_seen=self.end - 40 * DAY,
                last_seen=self.end - 40 * DAY,
                status="closed",
            ),
            cluster(
                "c4",
                first_seen=self.end - 2 * DAY,
                last_seen=self.end - 2 * DAY,
                membership_revision=1,
                summarized_revision=0,
            ),
        ]
        self.memberships = [
            membership("c1", "a1", self.end - 5 * DAY, "source-a"),
            membership("c1", "a2", self.end - DAY, "source-b"),
            membership("c2", "b1", self.end - 20 * DAY, "source-a"),
            membership("c3", "c1", self.end - 40 * DAY, "source-b"),
            membership("c4", "d1", self.end - 2 * DAY, "source-a"),
        ]

    def test_metrics_count_events_not_articles_and_compare_periods(self):
        metrics = compute_landscape_metrics(
            profile_id="profile",
            period_end_ts=self.end,
            clusters=self.clusters,
            memberships=self.memberships,
            settings=LandscapeMetricSettings(windows_days=(30,)),
        )
        window = metrics["windows"][0]

        self.assertEqual(3, window["independent_event_count"])
        self.assertEqual(4, window["article_count"])
        self.assertEqual(3, window["new_event_count"])
        self.assertEqual(1, window["reopened_event_count"])
        self.assertEqual(1, window["status_counts"]["dormant"])
        self.assertEqual(1, window["comparison"]["independent_event_count"])
        self.assertEqual(2, window["comparison"]["independent_event_delta"])
        self.assertEqual(0.75, window["source_coverage"]["dominant_source_share"])
        self.assertEqual(2, window["source_coverage"]["unique_sources_per_event"]["c1"])
        self.assertEqual(0.75, window["comparison"]["source_mix_distance"])
        self.assertIn("pending_cluster_snapshots", window["coverage"]["warnings"])
        self.assertFalse(window["coverage"]["trend_eligible"])

    def test_output_is_independent_of_input_order_and_duplicates_are_ignored(self):
        settings = LandscapeMetricSettings(windows_days=(30,))
        first = compute_landscape_metrics(
            profile_id="profile",
            period_end_ts=self.end,
            clusters=self.clusters,
            memberships=self.memberships,
            settings=settings,
        )
        duplicated = list(reversed(copy.deepcopy(self.memberships)))
        duplicated.append(copy.deepcopy(self.memberships[0]))
        second = compute_landscape_metrics(
            profile_id="profile",
            period_end_ts=self.end,
            clusters=list(reversed(copy.deepcopy(self.clusters))),
            memberships=duplicated,
            settings=settings,
        )

        self.assertEqual(first["windows"], second["windows"])
        self.assertEqual(1, second["quality"]["duplicate_membership_count"])
        self.assertIn("duplicate_memberships_ignored", second["quality"]["warning_codes"])

    def test_trend_requires_events_across_buckets_and_complete_snapshots(self):
        clusters = copy.deepcopy(self.clusters)
        clusters[3]["summarized_revision"] = 1
        memberships = copy.deepcopy(self.memberships)
        memberships[2]["article_ts"] = self.end - 10 * DAY
        memberships[2]["evidence"]["published_ts"] = self.end - 10 * DAY
        memberships[3]["article_ts"] = self.end - 20 * DAY
        memberships[3]["evidence"]["published_ts"] = self.end - 20 * DAY
        metrics = compute_landscape_metrics(
            profile_id="profile",
            period_end_ts=self.end,
            clusters=clusters,
            memberships=memberships,
            settings=LandscapeMetricSettings(
                windows_days=(14,),
                min_independent_events_for_trend=2,
                source_concentration_threshold=0.9,
                source_mix_change_threshold=0.9,
            ),
        )

        window = metrics["windows"][0]
        self.assertEqual(2, window["coverage"]["nonempty_time_bucket_count"])
        self.assertEqual(0, window["coverage"]["missing_time_bucket_count"])
        self.assertTrue(window["coverage"]["trend_eligible"])

    def test_atomic_singletons_do_not_create_snapshot_backlog_when_excluded(self):
        metrics = compute_landscape_metrics(
            profile_id="profile",
            period_end_ts=self.end,
            clusters=self.clusters,
            memberships=self.memberships,
            settings=LandscapeMetricSettings(
                windows_days=(30,),
                min_snapshot_member_count=2,
            ),
        )

        window = metrics["windows"][0]
        self.assertEqual(3, window["independent_event_count"])
        self.assertEqual(["c1"], window["snapshot_eligible_cluster_ids"])
        self.assertEqual(2, window["atomic_observation_cluster_count"])
        self.assertEqual(0, window["coverage"]["pending_snapshot_count"])
        self.assertNotIn(
            "pending_cluster_snapshots", window["coverage"]["warnings"]
        )

    def test_store_builder_falls_back_to_retained_article_source(self):
        rows = [membership("c1", "a1", self.end - DAY)]
        store = FakeStore(
            [self.clusters[0]],
            rows,
            articles=[{"id": "a1", "source": "retained-source"}],
        )
        metrics = build_landscape_metrics(
            store,
            profile_id="profile",
            period_end_ts=self.end,
            settings=LandscapeMetricSettings(windows_days=(7,)),
        )

        coverage = metrics["windows"][0]["source_coverage"]
        self.assertEqual({"retained-source": 1}, coverage["source_article_counts"])
        self.assertEqual(0, coverage["unknown_source_article_count"])

    def test_unresolved_and_excluded_reviews_do_not_affect_event_metrics(self):
        clusters = [
            cluster(
                "included",
                first_seen=self.end - DAY,
                last_seen=self.end - DAY,
            ),
            cluster(
                "unresolved",
                first_seen=self.end - DAY,
                last_seen=self.end - DAY,
                status="needs_review",
            ),
            {
                **cluster(
                    "excluded",
                    first_seen=self.end - DAY,
                    last_seen=self.end - DAY,
                    status="closed",
                ),
                "review_decision": "exclude",
            },
        ]
        memberships = [
            membership(row["id"], f"article-{row['id']}", self.end - DAY, "source")
            for row in clusters
        ]

        metrics = compute_landscape_metrics(
            profile_id="profile",
            period_end_ts=self.end,
            clusters=clusters,
            memberships=memberships,
            settings=LandscapeMetricSettings(windows_days=(7,)),
        )

        self.assertEqual(1, metrics["windows"][0]["independent_event_count"])
        self.assertEqual(1, metrics["quality"]["unresolved_review_cluster_count"])
        self.assertEqual(1, metrics["quality"]["excluded_review_cluster_count"])
        self.assertEqual(2, metrics["quality"]["ignored_review_membership_count"])
        self.assertIn(
            "unresolved_cluster_reviews_ignored",
            metrics["quality"]["warning_codes"],
        )


if __name__ == "__main__":
    unittest.main()
