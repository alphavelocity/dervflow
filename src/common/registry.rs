// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Model registry for dynamic registration and lookup of models

use crate::common::error::{DervflowError, Result};
use crate::common::traits::{Interpolator, Optimizer, PricingModel, StochasticProcess};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// Type aliases to reduce complexity
type PricingModelMap = Arc<RwLock<HashMap<String, Box<dyn PricingModel>>>>;
type ProcessMap = Arc<RwLock<HashMap<String, Box<dyn StochasticProcess>>>>;
type InterpolatorMap = Arc<RwLock<HashMap<String, Box<dyn Interpolator>>>>;
type OptimizerMap = Arc<RwLock<HashMap<String, Box<dyn Optimizer>>>>;

/// Registry for dynamically registering and retrieving models
///
/// This allows users to register custom implementations of pricing models,
/// stochastic processes, interpolators, and optimizers at runtime.
pub struct ModelRegistry {
    pricing_models: PricingModelMap,
    processes: ProcessMap,
    interpolators: InterpolatorMap,
    optimizers: OptimizerMap,
}

impl ModelRegistry {
    /// Create a new empty ModelRegistry
    pub fn new() -> Self {
        Self {
            pricing_models: Arc::new(RwLock::new(HashMap::new())),
            processes: Arc::new(RwLock::new(HashMap::new())),
            interpolators: Arc::new(RwLock::new(HashMap::new())),
            optimizers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a pricing model
    ///
    /// # Arguments
    /// * `model` - The pricing model to register
    ///
    /// # Returns
    /// * `Result<()>` - Success or error if registration fails
    pub fn register_pricing_model(&self, model: Box<dyn PricingModel>) -> Result<()> {
        let name = model.model_name().to_string();
        let mut models = self.pricing_models.write().map_err(|e| {
            DervflowError::DataError(format!("Failed to acquire write lock: {}", e))
        })?;

        if models.contains_key(&name) {
            return Err(DervflowError::InvalidInput(format!(
                "Pricing model '{}' is already registered",
                name
            )));
        }

        models.insert(name, model);
        Ok(())
    }

    /// Get a reference to a registered pricing model
    ///
    /// # Arguments
    /// * `name` - The name of the pricing model
    ///
    /// # Returns
    /// * `Result<PricingModelMap>` - The model or an error if not found
    pub fn get_pricing_model(&self, name: &str) -> Result<PricingModelMap> {
        let models = self
            .pricing_models
            .read()
            .map_err(|e| DervflowError::DataError(format!("Failed to acquire read lock: {}", e)))?;

        if !models.contains_key(name) {
            return Err(DervflowError::DataError(format!(
                "Pricing model '{}' not found in registry",
                name
            )));
        }

        Ok(Arc::clone(&self.pricing_models))
    }

    /// List all registered pricing model names
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - List of model names or an error
    pub fn list_pricing_models(&self) -> Result<Vec<String>> {
        let models = self
            .pricing_models
            .read()
            .map_err(|e| DervflowError::DataError(format!("Failed to acquire read lock: {}", e)))?;
        Ok(models.keys().cloned().collect())
    }

    /// Register a stochastic process
    ///
    /// # Arguments
    /// * `process` - The stochastic process to register
    ///
    /// # Returns
    /// * `Result<()>` - Success or error if registration fails
    pub fn register_process(&self, process: Box<dyn StochasticProcess>) -> Result<()> {
        let name = process.process_name().to_string();
        let mut processes = self.processes.write().map_err(|e| {
            DervflowError::DataError(format!("Failed to acquire write lock: {}", e))
        })?;

        if processes.contains_key(&name) {
            return Err(DervflowError::InvalidInput(format!(
                "Stochastic process '{}' is already registered",
                name
            )));
        }

        processes.insert(name, process);
        Ok(())
    }

    /// Get a reference to a registered stochastic process
    ///
    /// # Arguments
    /// * `name` - The name of the stochastic process
    ///
    /// # Returns
    /// * `Result<ProcessMap>` - The process or an error if not found
    pub fn get_process(&self, name: &str) -> Result<ProcessMap> {
        let processes = self
            .processes
            .read()
            .map_err(|e| DervflowError::DataError(format!("Failed to acquire read lock: {}", e)))?;

        if !processes.contains_key(name) {
            return Err(DervflowError::DataError(format!(
                "Stochastic process '{}' not found in registry",
                name
            )));
        }

        Ok(Arc::clone(&self.processes))
    }

    /// List all registered stochastic process names
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - List of process names or an error
    pub fn list_processes(&self) -> Result<Vec<String>> {
        let processes = self
            .processes
            .read()
            .map_err(|e| DervflowError::DataError(format!("Failed to acquire read lock: {}", e)))?;
        Ok(processes.keys().cloned().collect())
    }

    /// Register an interpolator
    ///
    /// # Arguments
    /// * `interpolator` - The interpolator to register
    ///
    /// # Returns
    /// * `Result<()>` - Success or error if registration fails
    pub fn register_interpolator(&self, interpolator: Box<dyn Interpolator>) -> Result<()> {
        let name = interpolator.interpolator_name().to_string();
        let mut interpolators = self.interpolators.write().map_err(|e| {
            DervflowError::DataError(format!("Failed to acquire write lock: {}", e))
        })?;

        if interpolators.contains_key(&name) {
            return Err(DervflowError::InvalidInput(format!(
                "Interpolator '{}' is already registered",
                name
            )));
        }

        interpolators.insert(name, interpolator);
        Ok(())
    }

    /// List all registered interpolator names
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - List of interpolator names or an error
    pub fn list_interpolators(&self) -> Result<Vec<String>> {
        let interpolators = self
            .interpolators
            .read()
            .map_err(|e| DervflowError::DataError(format!("Failed to acquire read lock: {}", e)))?;
        Ok(interpolators.keys().cloned().collect())
    }

    /// Register an optimizer
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to register
    ///
    /// # Returns
    /// * `Result<()>` - Success or error if registration fails
    pub fn register_optimizer(&self, optimizer: Box<dyn Optimizer>) -> Result<()> {
        let name = optimizer.optimizer_name().to_string();
        let mut optimizers = self.optimizers.write().map_err(|e| {
            DervflowError::DataError(format!("Failed to acquire write lock: {}", e))
        })?;

        if optimizers.contains_key(&name) {
            return Err(DervflowError::InvalidInput(format!(
                "Optimizer '{}' is already registered",
                name
            )));
        }

        optimizers.insert(name, optimizer);
        Ok(())
    }

    /// List all registered optimizer names
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - List of optimizer names or an error
    pub fn list_optimizers(&self) -> Result<Vec<String>> {
        let optimizers = self
            .optimizers
            .read()
            .map_err(|e| DervflowError::DataError(format!("Failed to acquire read lock: {}", e)))?;
        Ok(optimizers.keys().cloned().collect())
    }

    /// Clear all registered models
    pub fn clear(&self) -> Result<()> {
        self.pricing_models
            .write()
            .map_err(|e| DervflowError::DataError(format!("Failed to acquire write lock: {}", e)))?
            .clear();
        self.processes
            .write()
            .map_err(|e| DervflowError::DataError(format!("Failed to acquire write lock: {}", e)))?
            .clear();
        self.interpolators
            .write()
            .map_err(|e| DervflowError::DataError(format!("Failed to acquire write lock: {}", e)))?
            .clear();
        self.optimizers
            .write()
            .map_err(|e| DervflowError::DataError(format!("Failed to acquire write lock: {}", e)))?
            .clear();
        Ok(())
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Ensure ModelRegistry is thread-safe
unsafe impl Send for ModelRegistry {}
unsafe impl Sync for ModelRegistry {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::OptionParams;

    // Mock implementations for testing
    struct MockPricingModel {
        name: String,
    }

    impl MockPricingModel {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    impl PricingModel for MockPricingModel {
        fn price(&self, params: &OptionParams) -> Result<f64> {
            Ok(params.spot * 0.1)
        }

        fn model_name(&self) -> &str {
            &self.name
        }
    }

    struct MockProcess {
        name: String,
    }

    impl MockProcess {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    impl StochasticProcess for MockProcess {
        fn drift(&self, _t: f64, _x: f64) -> f64 {
            0.05
        }

        fn diffusion(&self, _t: f64, _x: f64) -> f64 {
            0.2
        }

        fn simulate_step(&self, _t: f64, x: f64, dt: f64, dw: f64) -> f64 {
            x + 0.05 * x * dt + 0.2 * x * dw
        }

        fn process_name(&self) -> &str {
            &self.name
        }
    }

    struct MockInterpolator {
        name: String,
    }

    impl MockInterpolator {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    impl Interpolator for MockInterpolator {
        fn interpolate(&self, x: f64) -> Result<f64> {
            Ok(x * 2.0)
        }

        fn fit(&mut self, _x: &[f64], _y: &[f64]) -> Result<()> {
            Ok(())
        }

        fn interpolator_name(&self) -> &str {
            &self.name
        }
    }

    struct MockOptimizer {
        name: String,
    }

    impl MockOptimizer {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    impl Optimizer for MockOptimizer {
        fn optimize(
            &self,
            _objective: &dyn Fn(&[f64]) -> f64,
            initial: &[f64],
        ) -> Result<Vec<f64>> {
            Ok(initial.to_vec())
        }

        fn optimizer_name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_register_and_list_pricing_models() {
        let registry = ModelRegistry::new();

        let model1 = Box::new(MockPricingModel::new("model1"));
        let model2 = Box::new(MockPricingModel::new("model2"));

        assert!(registry.register_pricing_model(model1).is_ok());
        assert!(registry.register_pricing_model(model2).is_ok());

        let models = registry.list_pricing_models().unwrap();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&"model1".to_string()));
        assert!(models.contains(&"model2".to_string()));
    }

    #[test]
    fn test_duplicate_pricing_model_registration() {
        let registry = ModelRegistry::new();

        let model1 = Box::new(MockPricingModel::new("model1"));
        let model2 = Box::new(MockPricingModel::new("model1")); // Same name

        assert!(registry.register_pricing_model(model1).is_ok());
        assert!(registry.register_pricing_model(model2).is_err());
    }

    #[test]
    fn test_get_pricing_model() {
        let registry = ModelRegistry::new();

        let model = Box::new(MockPricingModel::new("test_model"));
        assert!(registry.register_pricing_model(model).is_ok());

        assert!(registry.get_pricing_model("test_model").is_ok());
        assert!(registry.get_pricing_model("nonexistent").is_err());
    }

    #[test]
    fn test_register_and_list_processes() {
        let registry = ModelRegistry::new();

        let process1 = Box::new(MockProcess::new("gbm"));
        let process2 = Box::new(MockProcess::new("ou"));

        assert!(registry.register_process(process1).is_ok());
        assert!(registry.register_process(process2).is_ok());

        let processes = registry.list_processes().unwrap();
        assert_eq!(processes.len(), 2);
        assert!(processes.contains(&"gbm".to_string()));
        assert!(processes.contains(&"ou".to_string()));
    }

    #[test]
    fn test_register_and_list_interpolators() {
        let registry = ModelRegistry::new();

        let interp1 = Box::new(MockInterpolator::new("linear"));
        let interp2 = Box::new(MockInterpolator::new("cubic"));

        assert!(registry.register_interpolator(interp1).is_ok());
        assert!(registry.register_interpolator(interp2).is_ok());

        let interpolators = registry.list_interpolators().unwrap();
        assert_eq!(interpolators.len(), 2);
        assert!(interpolators.contains(&"linear".to_string()));
        assert!(interpolators.contains(&"cubic".to_string()));
    }

    #[test]
    fn test_register_and_list_optimizers() {
        let registry = ModelRegistry::new();

        let opt1 = Box::new(MockOptimizer::new("bfgs"));
        let opt2 = Box::new(MockOptimizer::new("nelder_mead"));

        assert!(registry.register_optimizer(opt1).is_ok());
        assert!(registry.register_optimizer(opt2).is_ok());

        let optimizers = registry.list_optimizers().unwrap();
        assert_eq!(optimizers.len(), 2);
        assert!(optimizers.contains(&"bfgs".to_string()));
        assert!(optimizers.contains(&"nelder_mead".to_string()));
    }

    #[test]
    fn test_clear_registry() {
        let registry = ModelRegistry::new();

        let model = Box::new(MockPricingModel::new("model"));
        let process = Box::new(MockProcess::new("process"));
        let interp = Box::new(MockInterpolator::new("interp"));
        let opt = Box::new(MockOptimizer::new("opt"));

        assert!(registry.register_pricing_model(model).is_ok());
        assert!(registry.register_process(process).is_ok());
        assert!(registry.register_interpolator(interp).is_ok());
        assert!(registry.register_optimizer(opt).is_ok());

        assert!(registry.clear().is_ok());

        assert_eq!(registry.list_pricing_models().unwrap().len(), 0);
        assert_eq!(registry.list_processes().unwrap().len(), 0);
        assert_eq!(registry.list_interpolators().unwrap().len(), 0);
        assert_eq!(registry.list_optimizers().unwrap().len(), 0);
    }

    #[test]
    fn test_registry_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ModelRegistry>();
    }
}
