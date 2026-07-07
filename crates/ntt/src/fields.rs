use ark_ff::define_field;

// BabyBear: p = 2^31 - 2^27 + 1, TWO_ADICITY = 27
define_field!(
    modulus = "2013265921",
    generator = "31",
    name = BabyBear,
);

// KoalaBear: p = 2^31 - 2^24 + 1, TWO_ADICITY = 24
define_field!(
    modulus = "2130706433",
    generator = "3",
    name = KoalaBear,
);

// Goldilocks: p = 2^64 - 2^32 + 1, TWO_ADICITY = 32
define_field!(
    modulus = "18446744069414584321",
    generator = "7",
    name = Goldilocks,
);

// TeddyBear: p = 2^32 - 2^30 + 1, TWO_ADICITY = 30
define_field!(
    modulus = "3221225473",
    generator = "5",
    name = TeddyBear,
);


#[cfg(feature = "babybear")]
pub type DefaultField = BabyBear;

#[cfg(feature = "koalabear")]
pub type DefaultField = KoalaBear;

#[cfg(feature = "goldilocks")]
pub type DefaultField = Goldilocks;

#[cfg(feature = "teddybear")]
pub type DefaultField = TeddyBear;

