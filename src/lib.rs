extern crate num_traits;
extern crate serde;
#[macro_use]
extern crate serde_derive;

use std::f64;
use std::ops::{Add,Mul,Div,Sub,Neg};
use std::cmp::Ordering;
use std::fmt;

use num_traits::{Zero,Float,Num,zero};
use num_traits::{FromPrimitive, ToPrimitive};

pub trait Angle<N>{
    fn pi() -> Self;
    fn two_pi() -> Self;
    fn half_pi() -> Self;
    fn to_rad(self) -> Rad<N>;
    fn to_deg(self) -> Deg<N>;
    fn wrap(self) -> Self;

    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn value(self) -> N;

    fn sin(self) -> N where N:Float;
    fn cos(self) -> N where N:Float;
    fn tan(self) -> N where N:Float;
    fn sin_cos(self) -> (N,N) where N:Float;
    fn abs(self) -> Self;
}

#[derive(Clone,Copy,Debug,Serialize,Deserialize)]
pub struct Deg<N>(pub N);

#[derive(Clone,Copy,Debug,Serialize,Deserialize)]
pub struct Rad<N>(pub N);

impl<N: Float + From<f64>> Angle<N> for Deg<N>{
	#[inline]
    fn to_rad(self) -> Rad<N>{
        Rad(self.value() * From::from(f64::consts::PI) / From::from(180.0))
    }

    #[inline]
    fn to_deg(self) -> Deg<N>{
        self
    }

	#[inline]
    fn pi() -> Deg<N>{
        Deg(From::from(180.0))
    }

    #[inline]
    fn two_pi() -> Deg<N>{
        Deg(From::from(360.0))
    }

    #[inline]
    fn half_pi() -> Deg<N>{
        Deg(From::from(90.0))
    }

	#[inline]
    fn wrap(self) -> Deg<N>{
		// algorithm from http://stackoverflow.com/a/5852628/599884
		self.clone() - Deg::two_pi() * Deg(From::from((self  / Deg::two_pi()).value()))
    }

    #[inline]
    fn max(self, other: Deg<N>) -> Deg<N>{
    	Deg(From::from(self.wrap().value().max(From::from(other.wrap().value()))))
    }

    #[inline]
    fn min(self, other: Deg<N>) -> Deg<N>{
    	Deg(From::from(self.wrap().value().min(From::from(other.wrap().value()))))
    }

    #[inline]
    fn value(self) -> N{
        self.0
    }

    #[inline]
    fn sin(self) -> N
    	where N: Float {
    	self.to_rad().value().sin()
    }

    #[inline]
    fn cos(self) -> N
    	where N: Float {
    	self.to_rad().value().cos()
    }

    #[inline]
    fn tan(self) -> N
    	where N: Float {
    	self.to_rad().value().tan()
    }

    #[inline]
    fn sin_cos(self) -> (N,N)
    	where N: Float {
    	self.to_rad().value().sin_cos()
    }

	#[inline]
    fn abs(self) -> Deg<N> {
    	Deg(From::from(self.0.abs()))
    }
}



impl<N: Float + From<f64>> Angle<N> for Rad<N>{
	#[inline]
    fn to_rad(self) -> Rad<N>{
        self
    }

    #[inline]
    fn to_deg(	self) -> Deg<N>{
        Deg(self.0 * From::from(180.0) / From::from(f64::consts::PI))
    }

	#[inline]
    fn pi() -> Rad<N>{
        Rad(From::from(f64::consts::PI))
    }

    #[inline]
    fn two_pi() -> Rad<N>{
        let pi: N = From::from(f64::consts::PI);
        Rad(pi * From::from(2.0))
    }

    #[inline]
    fn half_pi() -> Rad<N>{
        let pi: N = From::from(f64::consts::PI);
        Rad(pi * From::from(0.5))
    }

	#[inline]
    fn wrap(self) -> Rad<N>{
		// algorithm from http://stackoverflow.com/a/5852628/599884
		self - Rad::two_pi() * Rad((self  / Rad::two_pi()).value().floor())
    }

    #[inline]
    fn max(self, other: Rad<N>) -> Rad<N>{
    	Rad(self.wrap().value().max(other.wrap().value()))
    }

    #[inline]
    fn min(self, other: Rad<N>) -> Rad<N>{
    	Rad(self.wrap().value().min(other.wrap().value()))
    }

    #[inline]
    fn value(self) -> N{
        self.0
    }

    #[inline]
    fn sin(self) -> N {
    	self.value().sin()
    }

    #[inline]
    fn cos(self) -> N {
    	self.value().cos()
    }

    #[inline]
    fn tan(self) -> N {
    	self.value().tan()
    }

    #[inline]
    fn sin_cos(self) -> (N,N) {
    	self.value().sin_cos()
    }

	#[inline]
    fn abs(self) -> Rad<N> {
    	Rad(From::from(self.0.abs()))
    }
}

impl<N: Float + From<f64>> Zero for Deg<N>{
    /// Returns the additive identity.
    #[inline]
    fn zero() -> Deg<N>{
        Deg(zero())
    }

    #[inline]
    fn is_zero(&self) -> bool{
        *self == Deg::zero()
    }
}

impl<N: Float + From<f64>> Zero for Rad<N>{
    /// Returns the additive identity.
    #[inline]
    fn zero() -> Rad<N>{
        Rad(zero())
    }

    #[inline]
    fn is_zero(&self) -> bool{
        *self == Rad::zero()
    }
}

impl<N: Num> Add for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn add(self, other: Deg<N>) -> Deg<N>{
    	Deg(self.0 + other.0)
    }
}

impl<N: Float> Add for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn add(self, other: Rad<N>) -> Rad<N>{
    	Rad(self.0 + other.0)
    }
}

impl<N: Num> Sub for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn sub(self, other: Deg<N>) -> Deg<N>{
    	Deg(self.0 - other.0)
    }
}

impl<N: Float> Sub for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn sub(self, other: Rad<N>) -> Rad<N>{
    	Rad(self.0 - other.0)
    }
}

impl<N: Num> Mul for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn mul(self, other: Deg<N>) -> Deg<N>{
    	Deg(self.0 * other.0)
    }
}

impl<N: Float> Mul for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn mul(self, other: Rad<N>) -> Rad<N>{
    	Rad(self.0 * other.0)
    }
}

impl<N: Float> Mul<N> for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn mul(self, other: N) -> Deg<N>{
    	Deg(self.0 * other)
    }
}

impl<N: Float> Mul<N> for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn mul(self, other: N) -> Rad<N>{
    	Rad(self.0 * other)
    }
}

impl<N: Num> Div for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn div(self, other: Deg<N>) -> Deg<N>{
    	Deg(self.0 / other.0)
    }
}

impl<N: Float> Div for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn div(self, other: Rad<N>) -> Rad<N>{
    	Rad(self.0 / other.0)
    }
}

impl<N: Num> Div<N> for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn div(self, other: N) -> Deg<N>{
    	Deg(self.0 / other)
    }
}

impl<N: Float> Div<N> for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn div(self, other: N) -> Rad<N>{
    	Rad(self.0 / other)
    }
}

impl<N: Num + Neg<Output=N>> Neg for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn neg(self) -> Deg<N>{
    	Deg(-self.0)
    }
}

impl<N: Float> Neg for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn neg(self) -> Rad<N>{
    	Rad(-self.0)
    }
}

impl<N: Float + From<f64>> PartialEq for Deg<N>{

	#[inline]
	fn eq(&self, other: &Deg<N>) -> bool{
		self.clone().wrap().0.eq(&other.clone().wrap().0)
	}
}

impl<N: Float + From<f64>> PartialEq for Rad<N>{

	#[inline]
	fn eq(&self, other: &Rad<N>) -> bool{
		self.wrap().0.eq(&other.wrap().0)
	}
}

impl<N: Float + From<f64>> PartialOrd for Deg<N>{

	#[inline]
    fn partial_cmp(&self, other: &Deg<N>) -> Option<Ordering>{
    	self.clone().wrap().0.partial_cmp(&other.clone().wrap().0)
    }
}

impl<N: Float + From<f64>> PartialOrd for Rad<N>{

	#[inline]
    fn partial_cmp(&self, other: &Rad<N>) -> Option<Ordering>{
    	self.wrap().0.partial_cmp(&other.wrap().0)
    }
}

pub trait Cast<T> {
    fn from(t: T) -> Self;
}

impl<N1: From<N2>, N2: Float + From<f64>> From<Rad<N2>> for Deg<N1>{
	fn from(t: Rad<N2>) -> Deg<N1>{
		Deg(From::from(t.to_deg().0))
	}
}

impl<N1: From<N2>, N2: Float + From<f64>> From<Deg<N2>> for Rad<N1>{
	fn from(t: Deg<N2>) -> Rad<N1>{
		Rad(From::from(t.to_rad().0))
	}
}

impl<N: FromPrimitive> FromPrimitive for Deg<N>{
    fn from_i64(n: i64) -> Option<Deg<N>>{
        N::from_i64(n).map(|n| Deg(n))
    }

    fn from_u64(n: u64) -> Option<Deg<N>>{
        N::from_u64(n).map(|n| Deg(n))
    }
}

impl<N: FromPrimitive> FromPrimitive for Rad<N>{
    fn from_i64(n: i64) -> Option<Rad<N>>{
        N::from_i64(n).map(|n| Rad(n))
    }

    fn from_u64(n: u64) -> Option<Rad<N>>{
        N::from_u64(n).map(|n| Rad(n))
    }
}

impl<N: ToPrimitive> ToPrimitive for Deg<N>{
    fn to_i64(&self) -> Option<i64>{
        self.0.to_i64()
    }

    fn to_u64(&self) -> Option<u64>{
        self.0.to_u64()
    }
}

impl<N: ToPrimitive> ToPrimitive for Rad<N>{
    fn to_i64(&self) -> Option<i64>{
        self.0.to_i64()
    }

    fn to_u64(&self) -> Option<u64>{
        self.0.to_u64()
    }
}

impl<N: fmt::Display> fmt::Display for Deg<N>{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result{
        write!(f, "{} deg", self.0)
    }
}

impl<N: fmt::Display> fmt::Display for Rad<N>{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result{
        write!(f, "{} rad", self.0)
    }
}

#[test]
fn wrap64() {
	let pi: Rad<f64> = Rad::pi();
	let pi_wrapped = (pi + Rad::two_pi()).wrap();
	assert_eq!(pi, pi_wrapped)
}

#[test]
fn deg_eq_rad64() {
	let _180 = Deg(180.0f64);
	let pi = Rad::pi();
	assert_eq!(_180, pi.to_deg());
}

#[test]
fn deg_to_rad64() {
	let _180 = Deg(180.0f64);
	let pi = Rad(_180.to_rad().value());
	assert_eq!(_180, pi.to_deg());
}

#[test]
fn rad_to_deg64() {
	let pi: Rad<f64> = Rad::pi();
	let _180 = Deg(pi.to_deg().value());
	assert_eq!(_180, pi.to_deg());
}

#[test]
fn wrap32() {
	// use deg instead of rad, otherwise the float preccision is not enough
	let pi: Deg<f32> = Deg::pi();
	let pi_wrapped = (pi + Deg::two_pi()).wrap();
	assert_eq!(pi, pi_wrapped)

}

#[test]
fn deg_eq_rad32() {
	let _180 = Deg(180.0f32);
	let pi = Rad::pi();
	assert_eq!(_180, pi.to_deg());
}

#[test]
fn deg_to_rad32() {
	let _180 = Deg(180.0f32);
	let pi = Rad(_180.to_rad().value());
	assert_eq!(_180, pi.to_deg());
}

#[test]
fn rad_to_deg32() {
	let pi: Rad<f32> = Rad::pi();
	let _180 = Deg(pi.to_deg().value());
	assert_eq!(_180, pi.to_deg());
}

#[test]
fn cmp() {
	let _20 = Deg(20f32);
	let _370 = Deg(370f32);
	assert!(_20>_370,"{:?} shold be > {:?}");
	assert!(_370<_20,"{:?} shold be > {:?}");
}

#[test]
fn max() {
	let _20 = Deg(20f32);
	let _370 = Deg(370f32);
	assert_eq!(_20.max(_370),_20);
}

#[test]
fn min() {
	let _20 = Deg(20f32);
	let _370 = Deg(370f32);
	assert_eq!(_20.min(_370),_370);
}
