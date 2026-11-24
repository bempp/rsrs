#[cfg(feature = "hdf5-store")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rsrs_samples::hdf5_store::Hdf5SketchCache;

    let path = std::env::args()
        .nth(1)
        .ok_or("usage: validate_hdf5 <samples.h5>")?;
    let cache = Hdf5SketchCache::open(path)?;
    cache.validate_quick()?;
    let rows = cache.committed_rows().min(3);
    let _ = cache.read_block(0, rows)?;
    println!(
        "validated rows={} dimension={}",
        cache.committed_rows(),
        cache.metadata().operator_dimension
    );
    Ok(())
}

#[cfg(not(feature = "hdf5-store"))]
fn main() {
    panic!("enable the hdf5-store feature");
}
