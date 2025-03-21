
#[macro_export]
macro_rules! with_openblas_threads {
    ($body:expr) => {{
        use std::env;
        use rayon::current_num_threads;

        // Set OPENBLAS_NUM_THREADS to 1 before entering parallel execution
        env::set_var("OPENBLAS_NUM_THREADS", "1");

        let result = $body; // Execute the parallel code block

        // Restore to Rayon's thread count after the parallel region
        let num_threads = current_num_threads();
        env::set_var("OPENBLAS_NUM_THREADS", num_threads.to_string());

        result
    }};
}