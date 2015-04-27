# angle-rs

Rust angle wrapper to avoid ambiguous parameters + common operation over angles like wrapping, comparisons, arithmetic operations, trigonometric operations and conversions between rad and deg.

```rust
assert!(Deg(20)>Deg(370))
assert!(Deg(20).min(Deg(370)) == Deg(370))
assert_eq!(Deg(180.0f32).sin(),Rad::pi().sin()) 
```