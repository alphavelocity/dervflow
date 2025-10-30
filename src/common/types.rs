// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Common data types and structures used across the library
//!
//! This module provides the fundamental types used throughout dervflow for
//! representing financial instruments, market data, and option parameters.
//!
//! # Type Aliases
//!
//! The library uses type aliases for semantic clarity:
//! - [`Price`]: Asset prices and option values (f64)
//! - [`Rate`]: Interest rates and dividend yields (f64, annualized)
//! - [`Volatility`]: Volatility measures (f64, annualized)
//! - [`Time`]: Time periods (f64, in years)
//!
//! # Examples
//!
//! ## Creating Option Parameters
//!
//! ```rust
//! use dervflow::common::types::{OptionParams, OptionType};
//!
//! let params = OptionParams::new(
//!     100.0,  // spot price
//!     105.0,  // strike price
//!     0.05,   // risk-free rate (5%)
//!     0.02,   // dividend yield (2%)
//!     0.25,   // volatility (25%)
//!     0.5,    // time to maturity (6 months)
//!     OptionType::Call,
//! );
//!
//! // Validate parameters
//! assert!(params.validate().is_ok());
//! ```
//!
//! ## Working with Market Quotes
//!
//! ```rust
//! use dervflow::common::types::{OptionQuote, OptionType};
//!
//! let quote = OptionQuote::new(
//!     100.0,  // strike
//!     1.0,    // maturity (1 year)
//!     10.45,  // market price
//!     OptionType::Call,
//! );
//! ```

use std::fmt;

/// Asset price or option value (in currency units)
pub type Price = f64;

/// Interest rate or dividend yield (annualized, as decimal, e.g., 0.05 for 5%)
pub type Rate = f64;

/// Volatility measure (annualized, as decimal, e.g., 0.20 for 20%)
pub type Volatility = f64;

/// Time period (in years, e.g., 1.0 for one year, 0.5 for six months)
pub type Time = f64;

/// Option type: Call or Put
///
///  A call option gives the holder the right to buy the underlying asset,
/// while a put option gives the right to sell.
///
/// # Examples
///
/// ```rust
/// use dervflow::common::types::OptionType;
///
/// let call = OptionType::Call;
/// let put = OptionType::Put;
///
/// assert_eq!(format!("{}", call), "Call");
/// assert_eq!(format!("{}", put), "Put");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    /// Call option: right to buy the underlying asset
    Call,
    /// Put option: right to sell the underlying asset
    Put,
}

impl fmt::Display for OptionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptionType::Call => write!(f, "Call"),
            OptionType::Put => write!(f, "Put"),
        }
    }
}

/// Exercise style: European or American
///
/// Determines when an option can be exercised:
/// - European: only at maturity
/// - American: any time up to and including maturity
///
/// # Examples
///
/// ```rust
/// use dervflow::common::types::ExerciseStyle;
///
/// let european = ExerciseStyle::European;
/// let american = ExerciseStyle::American;
///
/// assert_eq!(format!("{}", european), "European");
/// assert_eq!(format!("{}", american), "American");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExerciseStyle {
    /// European style: can only be exercised at maturity
    European,
    /// American style: can be exercised any time before or at maturity
    American,
}

impl fmt::Display for ExerciseStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExerciseStyle::European => write!(f, "European"),
            ExerciseStyle::American => write!(f, "American"),
        }
    }
}

/// Parameters for option pricing
///
/// Contains all the necessary inputs for pricing vanilla options using
/// various models (Black-Scholes, binomial trees, Monte Carlo, etc.).
///
/// # Fields
///
/// * `spot` - Current price of the underlying asset
/// * `strike` - Strike/exercise price of the option
/// * `rate` - Risk-free interest rate (annualized, continuous compounding)
/// * `dividend` - Dividend yield (annualized, continuous compounding)
/// * `volatility` - Volatility of the underlying asset (annualized)
/// * `time_to_maturity` - Time until option expiration (in years)
/// * `option_type` - Call or Put
///
/// # Examples
///
/// ```rust
/// use dervflow::common::types::{OptionParams, OptionType};
///
/// // At-the-money call option
/// let atm_call = OptionParams::new(
///     100.0,  // spot = strike
///     100.0,
///     0.05,   // 5% risk-free rate
///     0.0,    // no dividends
///     0.20,   // 20% volatility
///     1.0,    // 1 year to maturity
///     OptionType::Call,
/// );
///
/// // Out-of-the-money put option
/// let otm_put = OptionParams::new(
///     100.0,  // spot
///     95.0,   // strike below spot
///     0.05,
///     0.02,   // 2% dividend yield
///     0.25,   // 25% volatility
///     0.5,    // 6 months to maturity
///     OptionType::Put,
/// );
///
/// // Validate parameters
/// assert!(atm_call.validate().is_ok());
/// assert!(otm_put.validate().is_ok());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct OptionParams {
    /// Current spot price of the underlying asset
    pub spot: Price,
    /// Strike price of the option
    pub strike: Price,
    /// Risk-free interest rate (annualized, continuous compounding)
    pub rate: Rate,
    /// Dividend yield (annualized, continuous compounding)
    pub dividend: Rate,
    /// Volatility of the underlying asset (annualized)
    pub volatility: Volatility,
    /// Time to maturity (in years)
    pub time_to_maturity: Time,
    /// Option type (Call or Put)
    pub option_type: OptionType,
}

impl OptionParams {
    /// Create a new OptionParams instance
    ///
    /// # Arguments
    ///
    /// * `spot` - Current price of the underlying asset (must be positive)
    /// * `strike` - Strike price of the option (must be positive)
    /// * `rate` - Risk-free interest rate (annualized, as decimal)
    /// * `dividend` - Dividend yield (annualized, as decimal)
    /// * `volatility` - Volatility (annualized, as decimal, must be non-negative)
    /// * `time_to_maturity` - Time to expiration in years (must be non-negative)
    /// * `option_type` - Call or Put
    ///
    /// # Examples
    ///
    /// ```rust
    /// use dervflow::common::types::{OptionParams, OptionType};
    ///
    /// let params = OptionParams::new(
    ///     100.0,  // $100 spot price
    ///     105.0,  // $105 strike price
    ///     0.05,   // 5% risk-free rate
    ///     0.02,   // 2% dividend yield
    ///     0.25,   // 25% volatility
    ///     1.0,    // 1 year to maturity
    ///     OptionType::Call,
    /// );
    /// ```
    pub fn new(
        spot: Price,
        strike: Price,
        rate: Rate,
        dividend: Rate,
        volatility: Volatility,
        time_to_maturity: Time,
        option_type: OptionType,
    ) -> Self {
        Self {
            spot,
            strike,
            rate,
            dividend,
            volatility,
            time_to_maturity,
            option_type,
        }
    }

    /// Validate that all parameters are within reasonable bounds
    ///
    /// Checks that:
    /// - Spot price is positive
    /// - Strike price is positive
    /// - Volatility is non-negative
    /// - Time to maturity is non-negative
    ///
    /// # Returns
    ///
    /// * `Ok(())` if all parameters are valid
    /// * `Err(String)` with a descriptive error message if validation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use dervflow::common::types::{OptionParams, OptionType};
    ///
    /// let valid = OptionParams::new(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
    /// assert!(valid.validate().is_ok());
    ///
    /// let invalid = OptionParams::new(-100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        if self.spot <= 0.0 {
            return Err(format!("Spot price must be positive, got {}", self.spot));
        }
        if self.strike <= 0.0 {
            return Err(format!(
                "Strike price must be positive, got {}",
                self.strike
            ));
        }
        if self.volatility < 0.0 {
            return Err(format!(
                "Volatility must be non-negative, got {}",
                self.volatility
            ));
        }
        if self.time_to_maturity < 0.0 {
            return Err(format!(
                "Time to maturity must be non-negative, got {}",
                self.time_to_maturity
            ));
        }
        Ok(())
    }
}

/// Market quote for an option
#[derive(Debug, Clone, Copy)]
pub struct OptionQuote {
    /// Strike price
    pub strike: Price,
    /// Time to maturity (in years)
    pub maturity: Time,
    /// Market price of the option
    pub price: Price,
    /// Option type (Call or Put)
    pub option_type: OptionType,
}

impl OptionQuote {
    /// Create a new OptionQuote instance
    pub fn new(strike: Price, maturity: Time, price: Price, option_type: OptionType) -> Self {
        Self {
            strike,
            maturity,
            price,
            option_type,
        }
    }
}

/// Market quote for a bond
#[derive(Debug, Clone, Copy)]
pub struct BondQuote {
    /// Time to maturity (in years)
    pub maturity: Time,
    /// Coupon rate (annualized)
    pub coupon: Rate,
    /// Market price of the bond
    pub price: Price,
    /// Payment frequency per year (e.g., 2 for semi-annual)
    pub frequency: u32,
}

impl BondQuote {
    /// Create a new BondQuote instance
    pub fn new(maturity: Time, coupon: Rate, price: Price, frequency: u32) -> Self {
        Self {
            maturity,
            coupon,
            price,
            frequency,
        }
    }
}

/// Financial instrument types
#[derive(Debug, Clone)]
pub enum Instrument {
    /// Option instrument with parameters
    Option(OptionParams),
    /// Stock instrument
    Stock {
        /// Stock ticker symbol
        ticker: String,
        /// Current spot price
        spot: Price,
    },
    /// Bond instrument
    Bond {
        /// Bond parameters
        bond_params: BondQuote,
    },
}

impl Instrument {
    /// Get the current market value of the instrument
    pub fn market_value(&self) -> Price {
        match self {
            Instrument::Option(_) => {
                // Option value would need to be calculated using a pricing model
                // This is a placeholder
                0.0
            }
            Instrument::Stock { spot, .. } => *spot,
            Instrument::Bond { bond_params } => bond_params.price,
        }
    }

    /// Get a string representation of the instrument type
    pub fn instrument_type(&self) -> &str {
        match self {
            Instrument::Option(_) => "Option",
            Instrument::Stock { .. } => "Stock",
            Instrument::Bond { .. } => "Bond",
        }
    }
}

/// Greeks (option sensitivities) for risk management
///
/// The Greeks measure the sensitivity of an option's price to changes in
/// various parameters. They are essential for risk management and hedging.
///
/// # Fields
///
/// * `delta` - Sensitivity to underlying price (∂V/∂S), range typically [-1, 1]
/// * `gamma` - Rate of change of delta (∂²V/∂S²), always non-negative
/// * `vega` - Sensitivity to volatility (∂V/∂σ), always non-negative
/// * `theta` - Time decay (∂V/∂t), typically negative for long positions
/// * `rho` - Sensitivity to interest rate (∂V/∂r)
///
/// # Interpretation
///
/// - **Delta**: For a $1 increase in spot price, option value changes by delta dollars
/// - **Gamma**: For a $1 increase in spot price, delta changes by gamma
/// - **Vega**: For a 1% increase in volatility, option value changes by vega dollars
/// - **Theta**: Option value decreases by theta dollars per day (when expressed daily)
/// - **Rho**: For a 1% increase in interest rate, option value changes by rho dollars
///
/// # Examples
///
/// ```rust
/// use dervflow::common::types::Greeks;
///
/// let greeks = Greeks::new(
///     0.5,    // delta: 50% hedge ratio
///     0.02,   // gamma: delta changes by 0.02 per $1 move
///     15.0,   // vega: $15 gain per 1% vol increase
///     -5.0,   // theta: loses $5 per year (time decay)
///     10.0,   // rho: $10 gain per 1% rate increase
/// );
///
/// println!("Delta: {:.4}", greeks.delta);
/// println!("Gamma: {:.4}", greeks.gamma);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Greeks {
    /// Delta: sensitivity to underlying price (∂V/∂S)
    pub delta: f64,
    /// Gamma: rate of change of delta (∂²V/∂S²)
    pub gamma: f64,
    /// Vega: sensitivity to volatility (∂V/∂σ)
    pub vega: f64,
    /// Theta: time decay (∂V/∂t)
    pub theta: f64,
    /// Rho: sensitivity to interest rate (∂V/∂r)
    pub rho: f64,
}

impl Greeks {
    /// Create a new Greeks instance with all values
    pub fn new(delta: f64, gamma: f64, vega: f64, theta: f64, rho: f64) -> Self {
        Self {
            delta,
            gamma,
            vega,
            theta,
            rho,
        }
    }

    /// Create a Greeks instance with all values set to zero
    pub fn zero() -> Self {
        Self {
            delta: 0.0,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        }
    }
}

/// Portfolio position holding an instrument and quantity
#[derive(Debug, Clone)]
pub struct Position {
    /// The financial instrument
    pub instrument: Instrument,
    /// Quantity held (positive for long, negative for short)
    pub quantity: f64,
    /// Current market value of the position
    pub market_value: Price,
}

impl Position {
    /// Create a new Position instance
    pub fn new(instrument: Instrument, quantity: f64) -> Self {
        let market_value = instrument.market_value() * quantity;
        Self {
            instrument,
            quantity,
            market_value,
        }
    }

    /// Update the market value based on current instrument value
    pub fn update_market_value(&mut self) {
        self.market_value = self.instrument.market_value() * self.quantity;
    }

    /// Check if the position is long (positive quantity)
    pub fn is_long(&self) -> bool {
        self.quantity > 0.0
    }

    /// Check if the position is short (negative quantity)
    pub fn is_short(&self) -> bool {
        self.quantity < 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_type_display() {
        assert_eq!(format!("{}", OptionType::Call), "Call");
        assert_eq!(format!("{}", OptionType::Put), "Put");
    }

    #[test]
    fn test_exercise_style_display() {
        assert_eq!(format!("{}", ExerciseStyle::European), "European");
        assert_eq!(format!("{}", ExerciseStyle::American), "American");
    }

    #[test]
    fn test_option_params_validation() {
        let valid_params = OptionParams::new(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
        assert!(valid_params.validate().is_ok());

        let invalid_spot = OptionParams::new(-100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
        assert!(invalid_spot.validate().is_err());

        let invalid_strike =
            OptionParams::new(100.0, -100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
        assert!(invalid_strike.validate().is_err());

        let invalid_volatility =
            OptionParams::new(100.0, 100.0, 0.05, 0.02, -0.2, 1.0, OptionType::Call);
        assert!(invalid_volatility.validate().is_err());

        let invalid_time = OptionParams::new(100.0, 100.0, 0.05, 0.02, 0.2, -1.0, OptionType::Call);
        assert!(invalid_time.validate().is_err());
    }

    #[test]
    fn test_option_quote_creation() {
        let quote = OptionQuote::new(100.0, 1.0, 10.5, OptionType::Call);
        assert_eq!(quote.strike, 100.0);
        assert_eq!(quote.maturity, 1.0);
        assert_eq!(quote.price, 10.5);
        assert_eq!(quote.option_type, OptionType::Call);
    }

    #[test]
    fn test_bond_quote_creation() {
        let bond = BondQuote::new(5.0, 0.05, 98.5, 2);
        assert_eq!(bond.maturity, 5.0);
        assert_eq!(bond.coupon, 0.05);
        assert_eq!(bond.price, 98.5);
        assert_eq!(bond.frequency, 2);
    }

    #[test]
    fn test_instrument_types() {
        let option_params = OptionParams::new(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
        let option_instrument = Instrument::Option(option_params);
        assert_eq!(option_instrument.instrument_type(), "Option");

        let stock_instrument = Instrument::Stock {
            ticker: "AAPL".to_string(),
            spot: 150.0,
        };
        assert_eq!(stock_instrument.instrument_type(), "Stock");
        assert_eq!(stock_instrument.market_value(), 150.0);

        let bond_params = BondQuote::new(5.0, 0.05, 98.5, 2);
        let bond_instrument = Instrument::Bond { bond_params };
        assert_eq!(bond_instrument.instrument_type(), "Bond");
        assert_eq!(bond_instrument.market_value(), 98.5);
    }

    #[test]
    fn test_position_creation() {
        let stock = Instrument::Stock {
            ticker: "AAPL".to_string(),
            spot: 150.0,
        };
        let position = Position::new(stock, 10.0);
        assert_eq!(position.quantity, 10.0);
        assert_eq!(position.market_value, 1500.0);
        assert!(position.is_long());
        assert!(!position.is_short());
    }

    #[test]
    fn test_position_short() {
        let stock = Instrument::Stock {
            ticker: "AAPL".to_string(),
            spot: 150.0,
        };
        let position = Position::new(stock, -5.0);
        assert_eq!(position.quantity, -5.0);
        assert_eq!(position.market_value, -750.0);
        assert!(!position.is_long());
        assert!(position.is_short());
    }
}
