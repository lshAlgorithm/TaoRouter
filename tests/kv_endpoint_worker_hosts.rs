use vllm_router_rs::kv_index::{parse_endpoint_mapping, resolve_endpoints};

const UNSPECIFIED_HOSTS: &[&str] = &[
    "0.0.0.0",
    "0",
    "0x0",
    "[::]",
    "[::ffff:0.0.0.0]",
    "[::ffff:0:0]",
];

#[test]
fn parser_rejects_unspecified_workers_with_valid_publishers() {
    for scheme in ["http", "https"] {
        for host in UNSPECIFIED_HOSTS {
            let mapping = format!("{scheme}://{host}:8000=tcp://publisher:5557");
            assert_eq!(
                parse_endpoint_mapping(&mapping),
                Err("KV connect host cannot be unspecified".into()),
                "{mapping}"
            );
        }
    }
}

#[test]
fn resolver_rejects_unspecified_workers_with_raw_valid_overrides() {
    for scheme in ["http", "https"] {
        for host in UNSPECIFIED_HOSTS {
            let workers = vec![format!("{scheme}://{host}:8000")];
            let mappings = vec![(workers[0].clone(), "tcp://publisher:5557".into())];
            assert_eq!(
                resolve_endpoints(&workers, &mappings, 5558),
                Err("KV connect host cannot be unspecified".into()),
                "{}",
                workers[0]
            );
        }
    }
}

#[test]
fn parser_rejects_wildcard_workers_with_valid_publishers() {
    for scheme in ["http", "https"] {
        let mapping = format!("{scheme}://*:8000=tcp://publisher:5557");
        assert_eq!(
            parse_endpoint_mapping(&mapping),
            Err("KV connect host cannot be a wildcard".into()),
            "{mapping}"
        );
    }
}

#[test]
fn resolver_rejects_wildcard_workers_with_raw_valid_overrides() {
    for scheme in ["http", "https"] {
        let workers = vec![format!("{scheme}://*:8000")];
        let mappings = vec![(workers[0].clone(), "tcp://publisher:5557".into())];
        assert_eq!(
            resolve_endpoints(&workers, &mappings, 5558),
            Err("KV connect host cannot be a wildcard".into()),
            "{}",
            workers[0]
        );
    }
}

#[test]
fn fallback_still_rejects_unspecified_and_wildcard_workers() {
    for scheme in ["http", "https"] {
        for host in UNSPECIFIED_HOSTS {
            assert_eq!(
                resolve_endpoints(&[format!("{scheme}://{host}:8000")], &[], 5557),
                Err("KV connect host cannot be unspecified".into()),
                "{scheme}://{host}"
            );
        }
        assert_eq!(
            resolve_endpoints(&[format!("{scheme}://*:8000")], &[], 5557),
            Err("KV connect host cannot be a wildcard".into())
        );
    }
}

#[test]
fn valid_workers_preserve_registry_keys_with_overrides_and_fallback() {
    for (worker, fallback) in [
        ("http://example.com", "tcp://example.com:5557"),
        ("https://example.com", "tcp://example.com:5557"),
        ("HTTP://EXAMPLE.COM:80/", "tcp://example.com:5557"),
        ("HTTPS://EXAMPLE.COM:443/", "tcp://example.com:5557"),
        ("http://127.1:8000/", "tcp://127.0.0.1:5557"),
        ("http://[::1]:8000/", "tcp://[::1]:5557"),
        (
            "HTTP://[0:0:0:0:0:FFFF:7F00:1]:8000/",
            "tcp://127.0.0.1:5557",
        ),
    ] {
        let workers = vec![worker.to_string()];
        let mapping = parse_endpoint_mapping(&format!("{worker}=tcp://publisher:5558")).unwrap();
        assert_eq!(
            resolve_endpoints(&workers, &[mapping], 5557).unwrap(),
            vec![(worker.to_string(), "tcp://publisher:5558".into())]
        );
        assert_eq!(
            resolve_endpoints(&workers, &[], 5557).unwrap(),
            vec![(worker.to_string(), fallback.into())]
        );
    }
}

#[test]
fn parser_rejects_worker_port_zero_with_valid_publisher() {
    for scheme in ["http", "https"] {
        assert_eq!(
            parse_endpoint_mapping(&format!("{scheme}://worker:0=tcp://publisher:5557")),
            Err("KV worker port must be nonzero".into())
        );
    }
}

#[test]
fn resolver_rejects_worker_port_zero_with_raw_override() {
    for scheme in ["http", "https"] {
        let worker = format!("{scheme}://worker:0");
        assert_eq!(
            resolve_endpoints(
                std::slice::from_ref(&worker),
                &[(worker.clone(), "tcp://publisher:5557".into())],
                5558,
            ),
            Err("KV worker port must be nonzero".into())
        );
    }
}

#[test]
fn fallback_rejects_worker_port_zero() {
    for scheme in ["http", "https"] {
        assert_eq!(
            resolve_endpoints(&[format!("{scheme}://worker:0")], &[], 5557),
            Err("KV worker port must be nonzero".into())
        );
    }
}

#[test]
fn ipv4_and_mapped_ipv6_workers_keep_distinct_identity_and_original_keys() {
    let workers = vec![
        "HTTP://127.0.0.1:8000/".to_string(),
        "HTTP://[0:0:0:0:0:FFFF:7F00:1]:8000/".to_string(),
    ];
    let mappings: Vec<_> = workers
        .iter()
        .zip(["tcp://127.0.0.1:5557", "tcp://[::ffff:127.0.0.1]:5558"])
        .map(|(worker, endpoint)| parse_endpoint_mapping(&format!("{worker}={endpoint}")).unwrap())
        .collect();
    assert_eq!(
        resolve_endpoints(&workers, &mappings, 5559).unwrap(),
        vec![
            (workers[0].clone(), "tcp://127.0.0.1:5557".into()),
            (workers[1].clone(), "tcp://127.0.0.1:5558".into()),
        ]
    );
}

#[test]
fn ipv4_and_mapped_ipv6_workers_cannot_share_normalized_publisher() {
    let workers = vec![
        "http://127.0.0.1:8000".to_string(),
        "http://[::ffff:127.0.0.1]:8000".to_string(),
    ];
    let mappings: Vec<_> = workers
        .iter()
        .zip(["tcp://127.0.0.1:5557", "tcp://[::ffff:127.0.0.1]:5557"])
        .map(|(worker, endpoint)| parse_endpoint_mapping(&format!("{worker}={endpoint}")).unwrap())
        .collect();
    assert_eq!(
        resolve_endpoints(&workers, &mappings, 5559),
        Err("workers cannot share KV endpoint tcp://127.0.0.1:5557; configure explicit per-worker endpoints".into())
    );
}

#[test]
fn unknown_worker_diagnostic_chooses_smallest_name_in_both_input_orders() {
    let workers = vec!["http://configured:8000".to_string()];
    let a = parse_endpoint_mapping("http://a-unknown:8000=tcp://publisher-a:5557").unwrap();
    let z = parse_endpoint_mapping("http://z-unknown:8000=tcp://publisher-z:5557").unwrap();
    for mappings in [[a.clone(), z.clone()], [z, a]] {
        assert_eq!(
            resolve_endpoints(&workers, &mappings, 5558),
            Err(
                "KV endpoint mapping refers to an unconfigured worker: http://a-unknown:8000"
                    .into()
            )
        );
    }
}
