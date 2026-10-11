use std::sync::Arc;

use vllm_router_rs::kv_index::{KvBlockIndexer, MatchQuery, ResidencyOwner, StorageTier};

#[test]
fn a_partial_compressed_edge_match_must_not_skip_to_its_child() {
    let index = KvBlockIndexer::new();
    let owner = ResidencyOwner::Worker {
        source: Arc::from("worker"),
        dp_rank: 0,
        incarnation: 0,
    };
    // Keep [x, y] compressed: no other store may split this edge before the query.
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        None,
        &[
            (Arc::from("s1"), Arc::from("x")),
            (Arc::from("s2"), Arc::from("y")),
        ],
    );
    index.store(
        0,
        owner,
        StorageTier::Device,
        Some(Arc::from("s2")),
        &[(Arc::from("s3"), Arc::from("z"))],
    );
    let hits = index.find_matches(&MatchQuery {
        group_idx: 0,
        local_hashes: vec![Arc::from("x"), Arc::from("z")],
        tiers_of_interest: vec![StorageTier::Device],
    });
    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].matched_depth, 1,
        "y was not matched; z is not contiguous"
    );
}

fn worker(source: &str, dp_rank: u32) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: Arc::from(source),
        dp_rank,
        incarnation: 0,
    }
}

fn store(
    index: &KvBlockIndexer,
    group_idx: u32,
    owner: &ResidencyOwner,
    tier: StorageTier,
    parent: Option<&str>,
    blocks: &[(&str, &str)],
) {
    index.store(
        group_idx,
        owner.clone(),
        tier,
        parent.map(Arc::from),
        &blocks
            .iter()
            .map(|(seq, local)| (Arc::from(*seq), Arc::from(*local)))
            .collect::<Vec<_>>(),
    );
}

fn query(group_idx: u32, hashes: &[&str], tiers: &[StorageTier]) -> MatchQuery {
    MatchQuery {
        group_idx,
        local_hashes: hashes.iter().map(|hash| Arc::from(*hash)).collect(),
        tiers_of_interest: tiers.to_vec(),
    }
}

fn compressed_prefix_with_child() -> KvBlockIndexer {
    let index = KvBlockIndexer::new();
    let owner = worker("worker", 0);
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        None,
        &[("s1", "x"), ("s2", "y")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("s2"),
        &[("s3", "z")],
    );
    index
}

#[test]
fn complete_and_query_exhausted_prefixes_keep_their_depth() {
    let index = compressed_prefix_with_child();
    for (hashes, expected_depth) in [
        (vec!["x", "y", "z"], 3),
        (vec!["x"], 1),
        (vec!["x", "y"], 2),
        (vec!["x", "z"], 1),
        (vec!["x", "y", "z", "missing"], 3),
    ] {
        let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
        assert_eq!(hits.len(), 1, "query {hashes:?}");
        assert_eq!(hits[0].matched_depth, expected_depth, "query {hashes:?}");
    }
}

#[test]
fn a_first_block_mismatch_has_no_positive_depth_match() {
    let index = compressed_prefix_with_child();
    for hashes in [vec![], vec!["missing"], vec!["y", "z"]] {
        assert!(
            index
                .find_matches(&query(0, &hashes, &[StorageTier::Device]))
                .is_empty(),
            "query {hashes:?}"
        );
    }
}

#[test]
fn a_deeper_compressed_edge_mismatch_keeps_only_the_contiguous_prefix() {
    let index = KvBlockIndexer::new();
    let owner = worker("worker", 0);
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        None,
        &[("sa", "a"), ("sb", "b")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("sb"),
        &[("sc", "c"), ("sd", "d")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("sd"),
        &[("se", "e")],
    );
    for (hashes, expected_depth) in [
        (vec!["a", "b", "c", "e"], 3),
        (vec!["a", "b", "e"], 2),
        (vec!["a", "b", "c"], 3),
        (vec!["a", "b", "c", "d", "e"], 5),
    ] {
        let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
        assert_eq!(hits.len(), 1, "query {hashes:?}");
        assert_eq!(hits[0].matched_depth, expected_depth, "query {hashes:?}");
    }
}

#[test]
fn repeated_local_hashes_do_not_skip_distinct_sequence_positions() {
    let index = KvBlockIndexer::new();
    let owner = worker("worker", 0);
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        None,
        &[("sa", "a"), ("sx1", "x"), ("sb", "b")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("sb"),
        &[("sx2", "x"), ("sc", "c")],
    );
    for (hashes, expected_depth) in [
        (vec!["a", "x", "x", "c"], 2),
        (vec!["a", "x", "b", "x", "c"], 5),
        (vec!["a", "x", "b", "x"], 4),
    ] {
        let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
        assert_eq!(hits.len(), 1, "query {hashes:?}");
        assert_eq!(hits[0].matched_depth, expected_depth, "query {hashes:?}");
    }
}

#[test]
fn a_real_alternate_branch_remains_matchable_by_its_owner() {
    let index = compressed_prefix_with_child();
    let alternate = worker("alternate", 0);
    // Shared x has the same engine identity; the diverging z has its own identity.
    store(
        &index,
        0,
        &alternate,
        StorageTier::Device,
        None,
        &[("s1", "x"), ("sxz", "z")],
    );
    for (hashes, expected_worker, expected_alternate) in
        [(vec!["x", "z"], 1, 2), (vec!["x", "y", "z"], 3, 1)]
    {
        let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
        assert_eq!(hits.len(), 2, "query {hashes:?}");
        for (source, expected_depth) in [
            ("worker", expected_worker),
            ("alternate", expected_alternate),
        ] {
            let hit = hits
                .iter()
                .find(|hit| hit.target.instance_id.as_ref() == source)
                .expect("each owner holds x");
            assert_eq!(hit.matched_depth, expected_depth, "{source}: {hashes:?}");
        }
    }
}

#[test]
fn target_rank_tier_and_group_claims_do_not_fill_each_others_gaps() {
    let index = compressed_prefix_with_child();
    let original = worker("worker", 0);
    let other_worker = worker("other", 0);
    let other_rank = worker("worker", 1);
    for (owner, tier) in [
        (&other_worker, StorageTier::Device),
        (&other_rank, StorageTier::Device),
        (&original, StorageTier::HostPinned),
    ] {
        store(&index, 0, owner, tier, None, &[("s1", "x")]);
        // This owner/tier has x and z, but no claim on the intervening y.
        store(&index, 0, owner, tier, Some("s2"), &[("s3", "z")]);
    }
    store(
        &index,
        1,
        &other_worker,
        StorageTier::Device,
        None,
        &[("s1", "x"), ("s2", "y"), ("s3", "z")],
    );
    let hits = index.find_matches(&query(
        0,
        &["x", "y", "z"],
        &[StorageTier::Device, StorageTier::HostPinned],
    ));
    assert_eq!(hits.len(), 4);
    for (owner, tier, expected_depth) in [
        (&original, StorageTier::Device, 3),
        (&other_worker, StorageTier::Device, 1),
        (&other_rank, StorageTier::Device, 1),
        (&original, StorageTier::HostPinned, 1),
    ] {
        let hit = hits
            .iter()
            .find(|hit| hit.target == owner.target() && hit.tier == tier)
            .expect("each target/tier holds x");
        assert_eq!(hit.matched_depth, expected_depth);
    }
    let host_hits = index.find_matches(&query(0, &["x", "y", "z"], &[StorageTier::HostPinned]));
    assert_eq!(host_hits.len(), 1);
    assert_eq!(host_hits[0].tier, StorageTier::HostPinned);
    assert_eq!(host_hits[0].matched_depth, 1);
    let group_hits = index.find_matches(&query(1, &["x", "y", "z"], &[StorageTier::Device]));
    assert_eq!(group_hits.len(), 1);
    assert_eq!(group_hits[0].target, other_worker.target());
    assert_eq!(group_hits[0].matched_depth, 3);
    assert!(index
        .find_matches(&query(2, &["x", "y", "z"], &[StorageTier::Device]))
        .is_empty());
}

#[test]
fn equivalent_long_short_and_mixed_edges_have_the_same_prefix_semantics() {
    let owner = worker("worker", 0);
    let path = [
        ("sa", "a"),
        ("sb", "b"),
        ("sc", "c"),
        ("sd", "d"),
        ("se", "e"),
        ("sf", "f"),
    ];
    let indexes: Vec<_> = [vec![4, 2], vec![1, 1, 1, 1, 1, 1], vec![2, 2, 2]]
        .into_iter()
        .map(|batch_lengths| {
            let index = KvBlockIndexer::new();
            let mut offset: usize = 0;
            for batch_length in batch_lengths {
                let parent = offset.checked_sub(1).map(|pos| path[pos].0);
                store(
                    &index,
                    0,
                    &owner,
                    StorageTier::Device,
                    parent,
                    &path[offset..offset + batch_length],
                );
                offset += batch_length;
            }
            store(
                &index,
                0,
                &owner,
                StorageTier::Device,
                None,
                &[("sa", "a"), ("sb", "b"), ("sq", "q")],
            );
            index
        })
        .collect();
    for (hashes, expected_depth) in [
        (vec![], 0),
        (vec!["missing", "a"], 0),
        (vec!["a"], 1),
        (vec!["a", "b"], 2),
        (vec!["a", "q"], 1),
        (vec!["a", "b", "q"], 3),
        (vec!["a", "b", "c", "e", "f"], 3),
        (vec!["a", "b", "c", "d", "e"], 5),
        (vec!["a", "b", "c", "d", "f"], 4),
        (vec!["a", "b", "c", "d", "e", "f"], 6),
        (vec!["a", "b", "c", "d", "e", "f", "missing"], 6),
    ] {
        for (shape, index) in indexes.iter().enumerate() {
            let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
            if expected_depth == 0 {
                assert!(hits.is_empty(), "shape {shape}, query {hashes:?}");
            } else {
                assert_eq!(hits.len(), 1, "shape {shape}, query {hashes:?}");
                assert_eq!(
                    hits[0].matched_depth, expected_depth,
                    "shape {shape}, query {hashes:?}"
                );
            }
        }
    }
}
