extern crate num_traits;
extern crate serde;
#[macro_use]
extern crate serde_derive;

use std::f64;
use std::ops::{Add,Mul,Div,Sub,Neg};
use std::ops::{AddAssign,MulAssign,DivAssign,SubAssign};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use num_traits::{Zero,Float,Num,Signed,zero,NumCast};
use num_traits::{FromPrimitive, ToPrimitive};

pub trait Angle<N>{
    fn pi() -> Self where N: NumCast;
    fn two_pi() -> Self where N: NumCast;
    fn half_pi() -> Self where N: NumCast;
    fn to_rad(self) -> Rad<N> where N: NumCast;
    fn to_deg(self) -> Deg<N> where N: NumCast;
    fn wrap(self) -> Self where N: NumCast + Clone;

    fn max(self, other: Self) -> Self where N: PartialOrd + NumCast + Clone;
    fn min(self, other: Self) -> Self where N: PartialOrd + NumCast + Clone;
    fn value(self) -> N;

    fn sin(self) -> N where N:Float + NumCast;
    fn cos(self) -> N where N:Float + NumCast;
    fn tan(self) -> N where N:Float + NumCast;
    fn sin_cos(self) -> (N,N) where N:Float + NumCast;
    fn abs(self) -> Self where N: Signed;
}

#[derive(Clone,Copy,Debug,Default,Serialize,Deserialize)]
pub struct Deg<N>(pub N);

#[derive(Clone,Copy,Debug,Default,Serialize,Deserialize)]
pub struct Rad<N>(pub N);

impl<N: Num> Angle<N> for Deg<N>{
	#[inline]
    fn to_rad(self) -> Rad<N> where N: NumCast{
        Rad(self.value() * num_traits::cast(f64::consts::PI).unwrap() / num_traits::cast(180.0).unwrap())
    }

    #[inline]
    fn to_deg(self) -> Deg<N>{
        self
    }

	#[inline]
    fn pi() -> Deg<N> where N: NumCast{
        Deg(num_traits::cast(180.0).unwrap())
    }

    #[inline]
    fn two_pi() -> Deg<N> where N: NumCast{
        Deg(num_traits::cast(360.0).unwrap())
    }

    #[inline]
    fn half_pi() -> Deg<N> where N: NumCast{
        Deg(num_traits::cast(90.0).unwrap())
    }

	#[inline]
    fn wrap(self) -> Deg<N> where N: NumCast + Clone{
		//self.clone() - Deg::<N>::two_pi() * Deg((self  / Deg::two_pi()).value().into())
        let selff: f64 = num_traits::cast(self.value()).unwrap();
        let norm: f64 = selff  / Deg::<f64>::two_pi().value();
        let floor: f64 = norm.floor();
        let v: N = num_traits::cast(selff - Deg::<f64>::two_pi().value() * floor).unwrap();
        Deg(v)
    }

    #[inline]
    fn max(self, other: Deg<N>) -> Deg<N> where N: PartialOrd + NumCast + Clone{
        let v = self.wrap().value();
        let o = other.wrap().value();
        if v > o { Deg(v) } else { Deg(o) }
    }

    #[inline]
    fn min(self, other: Deg<N>) -> Deg<N> where N: PartialOrd + NumCast + Clone{
        let v = self.wrap().value();
        let o = other.wrap().value();
        if v < o { Deg(v) } else { Deg(o) }
    }

    #[inline]
    fn value(self) -> N{
        self.0
    }

    #[inline]
    fn sin(self) -> N
    	where N: Float + NumCast {
    	self.to_rad().value().sin()
    }

    #[inline]
    fn cos(self) -> N
    	where N: Float + NumCast {
    	self.to_rad().value().cos()
    }

    #[inline]
    fn tan(self) -> N
    	where N: Float + NumCast {
    	self.to_rad().value().tan()
    }

    #[inline]
    fn sin_cos(self) -> (N,N)
    	where N: Float + NumCast {
    	self.to_rad().value().sin_cos()
    }

	#[inline]
    fn abs(self) -> Deg<N>
    	where N: Signed
    {
    	Deg(self.0.abs())
    }
}



impl<N:  Num> Angle<N> for Rad<N>{
	#[inline]
    fn to_rad(self) -> Rad<N>{
        self
    }

    #[inline]
    fn to_deg(	self) -> Deg<N> where N: NumCast{
        Deg(self.0 * num_traits::cast(180.0).unwrap() / num_traits::cast(f64::consts::PI).unwrap())
    }

	#[inline]
    fn pi() -> Rad<N> where N: NumCast{
        Rad(num_traits::cast(f64::consts::PI).unwrap())
    }

    #[inline]
    fn two_pi() -> Rad<N> where N: NumCast{
        let pi: N = num_traits::cast(f64::consts::PI).unwrap();
        Rad(pi * num_traits::cast(2.0).unwrap())
    }

    #[inline]
    fn half_pi() -> Rad<N> where N: NumCast{
        let pi: N = num_traits::cast(f64::consts::PI).unwrap();
        Rad(pi * num_traits::cast(0.5).unwrap())
    }

	#[inline]
    fn wrap(self) -> Rad<N> where N: NumCast + Clone {
		// self.clone() - Rad::<N>::two_pi() * Rad((self  / Rad::two_pi()).value().into())
        let selff: f64 = num_traits::cast(self.value()).unwrap();
        let norm: f64 = selff  / Rad::<f64>::two_pi().value();
        let floor: f64 = norm.floor();
        let v: N = num_traits::cast(selff - Rad::<f64>::two_pi().value() * floor).unwrap();
        Rad(v)
    }

    #[inline]
    fn max(self, other: Rad<N>) -> Rad<N> where N: PartialOrd + NumCast + Clone {
        let v = self.wrap().value();
        let o = other.wrap().value();
        if v > o { Rad(v) } else { Rad(o) }
    }

    #[inline]
    fn min(self, other: Rad<N>) -> Rad<N> where N: PartialOrd + NumCast + Clone {
        let v = self.wrap().value();
        let o = other.wrap().value();
        if v < o { Rad(v) } else { Rad(o) }
    }

    #[inline]
    fn value(self) -> N{
        self.0
    }

    #[inline]
    fn sin(self) -> N
    	where N: Float {
    	self.value().sin()
    }

    #[inline]
    fn cos(self) -> N
    	where N: Float {
    	self.value().cos()
    }

    #[inline]
    fn tan(self) -> N
    	where N: Float {
    	self.value().tan()
    }

    #[inline]
    fn sin_cos(self) -> (N,N)
    	where N: Float {
    	self.value().sin_cos()
    }

	#[inline]
    fn abs(self) -> Rad<N>  where N:Signed{
    	Rad(self.0.abs())
    }
}

impl<N: Num + Zero + NumCast + Clone> Zero for Deg<N>{
    /// Returns the additive identity.
    #[inline]
    fn zero() -> Deg<N>{
        Deg(zero())
    }

    #[inline]
    fn is_zero(&self) -> bool{
        self.clone().wrap().0 == N::zero()
    }
}

impl<N: Num + Zero + NumCast + Clone> Zero for Rad<N>{
    /// Returns the additive identity.
    #[inline]
    fn zero() -> Rad<N>{
        Rad(zero())
    }

    #[inline]
    fn is_zero(&self) -> bool{
        self.clone().wrap().0 == N::zero()
    }
}

impl<N: Add<N, Output = N>> Add for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn add(self, other: Deg<N>) -> Deg<N>{
    	Deg(self.0 + other.0)
    }
}

impl<N: Add<N, Output = N>> Add for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn add(self, other: Rad<N>) -> Rad<N>{
    	Rad(self.0 + other.0)
    }
}

impl<N: AddAssign<N>> AddAssign for Deg<N>{
    #[inline]
    fn add_assign(&mut self, other: Deg<N>){
    	self.0 += other.0
    }
}

impl<N: AddAssign<N>> AddAssign for Rad<N>{
    #[inline]
    fn add_assign(&mut self, other: Rad<N>){
    	self.0 += other.0
    }
}

impl<N: Sub<N, Output = N>> Sub for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn sub(self, other: Deg<N>) -> Deg<N>{
    	Deg(self.0 - other.0)
    }
}

impl<N: Sub<N, Output = N>> Sub for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn sub(self, other: Rad<N>) -> Rad<N>{
    	Rad(self.0 - other.0)
    }
}

impl<N: SubAssign<N>> SubAssign for Deg<N>{
    #[inline]
    fn sub_assign(&mut self, other: Deg<N>){
    	self.0 -= other.0
    }
}

impl<N: SubAssign<N>> SubAssign for Rad<N>{
    #[inline]
    fn sub_assign(&mut self, other: Rad<N>){
    	self.0 -= other.0
    }
}

impl<N: Mul<N, Output = N>> Mul for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn mul(self, other: Deg<N>) -> Deg<N>{
    	Deg(self.0 * other.0)
    }
}

impl<N: Mul<N, Output = N>> Mul for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn mul(self, other: Rad<N>) -> Rad<N>{
    	Rad(self.0 * other.0)
    }
}

impl<N: Mul<N, Output = N>> Mul<N> for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn mul(self, other: N) -> Deg<N>{
    	Deg(self.0 * other)
    }
}

impl<N: Mul<N, Output = N>> Mul<N> for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn mul(self, other: N) -> Rad<N>{
    	Rad(self.0 * other)
    }
}

impl<N: MulAssign<N>> MulAssign for Deg<N>{
    #[inline]
    fn mul_assign(&mut self, other: Deg<N>){
    	self.0 *= other.0
    }
}

impl<N: MulAssign<N>> MulAssign for Rad<N>{
    #[inline]
    fn mul_assign(&mut self, other: Rad<N>){
    	self.0 *= other.0
    }
}

impl<N: MulAssign<N>> MulAssign<N> for Deg<N>{
    #[inline]
    fn mul_assign(&mut self, other: N){
    	self.0 *= other
    }
}

impl<N: MulAssign<N>> MulAssign<N> for Rad<N>{
    #[inline]
    fn mul_assign(&mut self, other: N){
    	self.0 *= other
    }
}

impl<N: Div<N, Output = N>> Div for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn div(self, other: Deg<N>) -> Deg<N>{
    	Deg(self.0 / other.0)
    }
}

impl<N: Div<N, Output = N>> Div for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn div(self, other: Rad<N>) -> Rad<N>{
    	Rad(self.0 / other.0)
    }
}

impl<N: Div<N, Output = N>> Div<N> for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn div(self, other: N) -> Deg<N>{
    	Deg(self.0 / other)
    }
}

impl<N: Div<N, Output = N>> Div<N> for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn div(self, other: N) -> Rad<N>{
    	Rad(self.0 / other)
    }
}

impl<N: DivAssign<N>> DivAssign for Deg<N>{
    #[inline]
    fn div_assign(&mut self, other: Deg<N>){
    	self.0 /= other.0
    }
}

impl<N: DivAssign<N>> DivAssign for Rad<N>{
    #[inline]
    fn div_assign(&mut self, other: Rad<N>){
    	self.0 /= other.0
    }
}

impl<N: DivAssign<N>> DivAssign<N> for Deg<N>{
    #[inline]
    fn div_assign(&mut self, other: N){
    	self.0 /= other
    }
}

impl<N: DivAssign<N>> DivAssign<N> for Rad<N>{
    #[inline]
    fn div_assign(&mut self, other: N){
    	self.0 /= other
    }
}

impl<N: Neg<Output=N>> Neg for Deg<N>{
    type Output = Deg<N>;

    #[inline]
    fn neg(self) -> Deg<N>{
    	Deg(-self.0)
    }
}

impl<N: Neg<Output=N>> Neg for Rad<N>{
    type Output = Rad<N>;

    #[inline]
    fn neg(self) -> Rad<N>{
    	Rad(-self.0)
    }
}

impl<N: PartialEq + Num + Clone + NumCast> PartialEq for Deg<N>{

	#[inline]
	fn eq(&self, other: &Deg<N>) -> bool{
		self.clone().wrap().0.eq(&other.clone().wrap().0)
	}
}

impl<N: Eq + PartialEq + Num + Clone + NumCast> Eq for Deg<N>{}

impl<N: PartialEq + Num + Clone + NumCast> PartialEq for Rad<N>{

	#[inline]
	fn eq(&self, other: &Rad<N>) -> bool{
		self.clone().wrap().0.eq(&other.clone().wrap().0)
	}
}

impl<N: Eq + PartialEq + Num + Clone + NumCast> Eq for Rad<N>{}

impl<N: PartialOrd + Num + Clone + NumCast> PartialOrd for Deg<N>{

	#[inline]
    fn partial_cmp(&self, other: &Deg<N>) -> Option<Ordering>{
    	self.clone().wrap().0.partial_cmp(&other.clone().wrap().0)
    }
}

impl<N: Ord + PartialEq + Num + Clone + NumCast> Ord for Deg<N>{

	#[inline]
    fn cmp(&self, other: &Deg<N>) -> Ordering{
    	self.clone().wrap().0.cmp(&other.clone().wrap().0)
    }
}

impl<N: PartialOrd + Num + Clone + NumCast> PartialOrd for Rad<N>{

	#[inline]
    fn partial_cmp(&self, other: &Rad<N>) -> Option<Ordering>{
    	self.clone().wrap().0.partial_cmp(&other.clone().wrap().0)
    }
}

impl<N: Ord + PartialEq + Num + Clone + NumCast> Ord for Rad<N>{

	#[inline]
    fn cmp(&self, other: &Rad<N>) -> Ordering{
    	self.clone().wrap().0.cmp(&other.clone().wrap().0)
    }
}

impl<N1: From<N2>, N2: Num + Clone + NumCast> From<Rad<N2>> for Deg<N1>{
	fn from(t: Rad<N2>) -> Deg<N1>{
		Deg(t.to_deg().0.into())
	}
}

impl<N1: From<N2>, N2: Num + Clone + NumCast> From<Deg<N2>> for Rad<N1>{
	fn from(t: Deg<N2>) -> Rad<N1>{
		Rad(t.to_rad().0.into())
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

impl<N> Hash for Deg<N> where N: Hash{
    fn hash<H: Hasher>(&self, state: &mut H){
        self.0.hash(state)
    }
}

impl<N> Hash for Rad<N> where N: Hash{
    fn hash<H: Hasher>(&self, state: &mut H){
        self.0.hash(state)
    }
}

pub trait AngleCast{
    fn from<N: NumCast + Num, T: Angle<N>>(v: T) -> Option<Self> where Self: Sized;
}

impl<N: NumCast> AngleCast for Rad<N>{
    fn from<N2: NumCast + Num, T: Angle<N2>>(v: T) -> Option<Rad<N>>{
        num_traits::cast(v.to_rad().value()).map(|v| Rad(v))
    }
}

impl<N: NumCast> AngleCast for Deg<N>{
    fn from<N2: NumCast + Num, T: Angle<N2>>(v: T) -> Option<Deg<N>>{
        num_traits::cast(v.to_deg().value()).map(|v| Deg(v))
    }
}

pub fn cast<T,U,N1,N2>(t: T) -> Option<U>
where T: AngleCast + Angle<N1>,
      U: AngleCast + Angle<N2>,
      N1: NumCast + Num,
      N2: NumCast + Num,
{
    U::from(t)
}

#[test]
fn wrap64() {
	let pi: Rad<f64> = Rad::pi();
	let pi_wrapped = (pi + Rad::two_pi()).wrap();
	assert_eq!(pi, pi_wrapped)
}

#[test]
fn negative_wrap64() {
	let pi: Rad<f64> = Rad::pi();
	let pi_wrapped = (pi - Rad::two_pi()).wrap();
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
fn negative_wrap32() {
	// use deg instead of rad, otherwise the float preccision is not enough
	let _90neg: Deg<f32> = Deg(-90.);
    let _270: Deg<f32> = Deg(270.);
	assert_eq!(_90neg.wrap().value(), _270.value())
}

#[test]
fn negative_wrapunsigned() {
	// use deg instead of rad, otherwise the float preccision is not enough
	let _90neg: Deg<i32> = Deg(-90);
    let _270: Deg<i32> = Deg(270);
	assert_eq!(_90neg.wrap().value(), _270.value())
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

#[test]
fn default() {
    let deg: Deg<f64> = Default::default();
    assert_eq!(deg.0, 0f64);
    let rad: Rad<f64> = Default::default();
    assert_eq!(rad.0, 0f64);
}

#[cfg(feature="simba")]
impl<T: Copy, S: simba::scalar::SubsetOf<T>> simba::scalar::SubsetOf<Deg<T>> for Deg<S> {
    fn to_superset(&self) -> Deg<T> {
        Deg(self.0.to_superset())
    }

    fn from_superset_unchecked(element: &Deg<T>) -> Self {
        Deg(S::from_superset_unchecked(&element.0))
    }

    fn is_in_subset(element: &Deg<T>) -> bool {
        S::is_in_subset(&element.0)
    }
}

#[cfg(feature="simba")]
impl<T: Copy, S: simba::scalar::SubsetOf<T>> simba::scalar::SubsetOf<Rad<T>> for Rad<S> {
    fn to_superset(&self) -> Rad<T> {
        Rad(self.0.to_superset())
    }

    fn from_superset_unchecked(element: &Rad<T>) -> Self {
        Rad(S::from_superset_unchecked(&element.0))
    }

    fn is_in_subset(element: &Rad<T>) -> bool {
        S::is_in_subset(&element.0)
    }
}