use ff::{derive::subtle::CtOption, PrimeField};

pub mod bn256;
pub mod utils;
pub trait MontgomeryRepr: PrimeField + From<u64> {
    fn from_montgomery_repr(repr: Self::Repr) -> CtOption<Self>;
    fn to_montgomery_repr(&self) -> Self::Repr;
}