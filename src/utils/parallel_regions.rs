#[macro_export]
macro_rules! with_openblas_threads {
    ($body:expr, $num_threads:expr) => {{
        use rayon::current_num_threads;
        use std::env;

        // Set OPENBLAS_NUM_THREADS to 1 before entering parallel execution
        env::set_var("OPENBLAS_NUM_THREADS", $num_threads.to_string());

        let result = $body; // Execute the parallel code block

        // Restore to Rayon's thread count after the parallel region
        let num_threads = current_num_threads();
        env::set_var("OPENBLAS_NUM_THREADS", num_threads.to_string());

        result
    }};
}
