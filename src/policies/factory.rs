//! Factory for creating load balancing policies

use super::{
    CacheAwareConfig, CacheAwarePolicy, ConsistentHashPolicy, CostParams, KvAwarePolicy,
    LoadBalancingPolicy, PowerOfTwoPolicy, RandomPolicy, RendezvousHashPolicy, RoundRobinPolicy,
    TransferCredits,
};
use crate::config::PolicyConfig;
use std::sync::Arc;

/// Factory for creating policy instances
pub struct PolicyFactory;

impl PolicyFactory {
    /// Create a policy from configuration
    pub fn create_from_config(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        match config {
            PolicyConfig::TrustedKvAware { config } => {
                Arc::new(super::TrustedKvAwarePolicy::with_config(config.clone()))
            }
            PolicyConfig::Random => Arc::new(RandomPolicy::new()),
            PolicyConfig::RoundRobin => Arc::new(RoundRobinPolicy::new()),
            PolicyConfig::PowerOfTwo { .. } => Arc::new(PowerOfTwoPolicy::new()),
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                let config = CacheAwareConfig {
                    cache_threshold: *cache_threshold,
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    eviction_interval_secs: *eviction_interval_secs,
                    max_tree_size: *max_tree_size,
                };
                Arc::new(CacheAwarePolicy::with_config(config))
            }
            PolicyConfig::KvAware {
                router_temperature,
                kv_penalty_host,
                kv_penalty_external,
                kv_penalty_disk,
            } => Arc::new(KvAwarePolicy::new(CostParams {
                router_temperature: *router_temperature,
                transfer_credits: TransferCredits {
                    device: 1.0,
                    host_pinned: *kv_penalty_host,
                    external: *kv_penalty_external,
                    disk: *kv_penalty_disk,
                },
                ..CostParams::default()
            })),
            PolicyConfig::ConsistentHash { virtual_nodes: _ } => {
                // Note: virtual_nodes parameter is available but not currently used
                // The consistent hash policy uses a hardcoded value for now
                Arc::new(ConsistentHashPolicy::new())
            }
            PolicyConfig::RendezvousHash => Arc::new(RendezvousHashPolicy::new()),
        }
    }

    /// Create a policy by name (for dynamic loading)
    pub fn create_by_name(name: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        match name.to_lowercase().as_str() {
            "trusted_kv_aware" => Some(Arc::new(super::TrustedKvAwarePolicy::default())),
            "random" => Some(Arc::new(RandomPolicy::new())),
            "round_robin" | "roundrobin" => Some(Arc::new(RoundRobinPolicy::new())),
            "power_of_two" | "poweroftwo" => Some(Arc::new(PowerOfTwoPolicy::new())),
            "cache_aware" | "cacheaware" => Some(Arc::new(CacheAwarePolicy::new())),
            "kv_aware" | "kvaware" => Some(Arc::new(KvAwarePolicy::default())),
            "consistent_hash" | "consistenthash" => Some(Arc::new(ConsistentHashPolicy::new())),
            "rendezvous_hash" | "rendezvoushash" => Some(Arc::new(RendezvousHashPolicy::new())),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_policy_modes_keep_distinct_capabilities() {
        for (name, trusted) in [("kv_aware", false), ("trusted_kv_aware", true)] {
            let config: PolicyConfig =
                serde_json::from_value(serde_json::json!({"type": name})).unwrap();
            assert_eq!(config.name(), name);
            let policy = PolicyFactory::create_from_config(&config);
            assert_eq!(policy.name(), name);
            assert_eq!(
                policy.as_any().is::<super::super::TrustedKvAwarePolicy>(),
                trusted
            );
            assert_eq!(
                super::super::kv_aware_ops(policy.as_ref()).is_some(),
                !trusted
            );
            assert_eq!(PolicyFactory::create_by_name(name).unwrap().name(), name);
        }
    }

    #[test]
    fn test_trusted_limits_do_not_replace_upstream_cost_parameters() {
        let policy: PolicyConfig = serde_json::from_value(serde_json::json!({
            "type": "kv_aware", "router_temperature": 0.25, "kv_penalty_host": 0.6
        }))
        .unwrap();
        assert!(matches!(policy, PolicyConfig::KvAware {
            router_temperature, kv_penalty_host, ..
        } if router_temperature == 0.25 && kv_penalty_host == 0.6));

        let policy: PolicyConfig = serde_json::from_value(serde_json::json!({
            "type": "trusted_kv_aware", "cache_weight": 0.75, "max_sources": 32
        }))
        .unwrap();
        assert!(matches!(policy, PolicyConfig::TrustedKvAware { config }
            if config.cache_weight == 0.75 && config.max_sources == 32));
    }

    #[test]
    fn test_create_from_config() {
        // Test Random
        let policy = PolicyFactory::create_from_config(&PolicyConfig::Random);
        assert_eq!(policy.name(), "random");

        // Test RoundRobin
        let policy = PolicyFactory::create_from_config(&PolicyConfig::RoundRobin);
        assert_eq!(policy.name(), "round_robin");

        // Test PowerOfTwo
        let policy = PolicyFactory::create_from_config(&PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        });
        assert_eq!(policy.name(), "power_of_two");

        // Test CacheAware
        let policy = PolicyFactory::create_from_config(&PolicyConfig::CacheAware {
            cache_threshold: 0.7,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 30,
            max_tree_size: 1000,
        });
        assert_eq!(policy.name(), "cache_aware");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
        });
        assert_eq!(policy.name(), "kv_aware");

        // Test ConsistentHash
        let policy =
            PolicyFactory::create_from_config(&PolicyConfig::ConsistentHash { virtual_nodes: 160 });
        assert_eq!(policy.name(), "consistent_hash");

        // Test RendezvousHash
        let policy = PolicyFactory::create_from_config(&PolicyConfig::RendezvousHash);
        assert_eq!(policy.name(), "rendezvous_hash");
    }

    #[test]
    fn test_create_by_name() {
        assert!(PolicyFactory::create_by_name("random").is_some());
        assert!(PolicyFactory::create_by_name("RANDOM").is_some());
        assert!(PolicyFactory::create_by_name("round_robin").is_some());
        assert!(PolicyFactory::create_by_name("RoundRobin").is_some());
        assert!(PolicyFactory::create_by_name("power_of_two").is_some());
        assert!(PolicyFactory::create_by_name("PowerOfTwo").is_some());
        assert!(PolicyFactory::create_by_name("cache_aware").is_some());
        assert!(PolicyFactory::create_by_name("CacheAware").is_some());
        assert!(PolicyFactory::create_by_name("kv_aware").is_some());
        assert!(PolicyFactory::create_by_name("consistent_hash").is_some());
        assert!(PolicyFactory::create_by_name("ConsistentHash").is_some());
        assert!(PolicyFactory::create_by_name("rendezvous_hash").is_some());
        assert!(PolicyFactory::create_by_name("RendezvousHash").is_some());
        assert!(PolicyFactory::create_by_name("unknown").is_none());
    }
}
