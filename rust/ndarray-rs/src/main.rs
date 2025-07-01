mod matmul;
use matmul::{matrix_multiply_double, matrix_multiply_float};

use clap::{Arg, ArgAction, Command};
use rand::prelude::*;
use std::time::Instant;

fn initialize_matrix_float(n: usize, matrix: &mut [f32]) {
    let mut rng = rand::rng();
    for i in 0..n * n {
        matrix[i] = rng.random::<f32>();
    }
}

fn initialize_matrix_double(n: usize, matrix: &mut [f64]) {
    let mut rng = rand::rng();
    for i in 0..n * n {
        matrix[i] = rng.random::<f64>();
    }
}

fn execute_float(dim: usize, loop_num: usize) -> (f64, f64, f64, f64) {
    let mut ave_gflops: f64 = 0.0;
    let mut max_gflops: f64 = 0.0;
    let mut ave_time: f64 = 0.0;
    let mut min_time = f64::MAX;

    for i in 0..loop_num {
        let mut a = vec![0.0; dim * dim];
        let mut b = vec![0.0; dim * dim];
        let mut c = vec![0.0; dim * dim];

        initialize_matrix_float(dim, &mut a);
        initialize_matrix_float(dim, &mut b);

        let start = Instant::now();
        matrix_multiply_float(dim, &a, &b, &mut c);
        let cpu_time = start.elapsed().as_secs_f64();

        let gflops = 2.0 * (dim * dim * dim) as f64 / cpu_time / 1e9;
        println!(
            "{}\t: {} x {} Matrix multiply wall time : {:.6}s({:.3}Gflops)",
            i + 1,
            dim,
            dim,
            cpu_time,
            gflops
        );

        ave_gflops += gflops;
        max_gflops = max_gflops.max(gflops);
        ave_time += cpu_time;
        min_time = min_time.min(cpu_time);
    }

    ave_gflops /= loop_num as f64;
    ave_time /= loop_num as f64;
    (ave_gflops, max_gflops, ave_time, min_time)
}

fn execute_double(dim: usize, loop_num: usize) -> (f64, f64, f64, f64) {
    let mut ave_gflops: f64 = 0.0;
    let mut max_gflops: f64 = 0.0;
    let mut ave_time: f64 = 0.0;
    let mut min_time = f64::MAX;

    for i in 0..loop_num {
        let mut a = vec![0.0; dim * dim];
        let mut b = vec![0.0; dim * dim];
        let mut c = vec![0.0; dim * dim];

        initialize_matrix_double(dim, &mut a);
        initialize_matrix_double(dim, &mut b);

        let start = Instant::now();
        matrix_multiply_double(dim, &a, &b, &mut c);
        let cpu_time = start.elapsed().as_secs_f64();

        let gflops = 2.0 * (dim * dim * dim) as f64 / cpu_time / 1e9;
        println!(
            "{}\t: {} x {} Matrix multiply wall time : {:.6}s({:.3}Gflops)",
            i + 1,
            dim,
            dim,
            cpu_time,
            gflops
        );

        ave_gflops += gflops;
        max_gflops = max_gflops.max(gflops);
        ave_time += cpu_time;
        min_time = min_time.min(cpu_time);
    }

    ave_gflops /= loop_num as f64;
    ave_time /= loop_num as f64;
    (ave_gflops, max_gflops, ave_time, min_time)
}

fn main() {
    let matches = Command::new("ndarray-rs")
        .version("0.1.0")
        .author("AndrewMoa")
        .about("Matrix multiplication benchmark")
        .arg(
            Arg::new("size")
                .short('n')
                .long("size")
                .help("Matrix size exponent (size = 2^n)")
                .default_value("10"),
        )
        .arg(
            Arg::new("loops")
                .short('l')
                .long("loops")
                .help("Number of iterations")
                .default_value("5"),
        )
        .arg(
            Arg::new("f64")
                .short('d')
                .long("f64")
                .help("Use float64 precision")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("f32")
                .short('f')
                .long("f32")
                .help("Use float32 precision (default)")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    let n: usize = matches
        .get_one::<String>("size")
        .unwrap()
        .parse()
        .expect("Invalid size exponent");
    let loop_num: usize = matches
        .get_one::<String>("loops")
        .unwrap()
        .parse()
        .expect("Invalid loop count");
    let use_double = matches.get_flag("f64");
    let use_float = matches.get_flag("f32");

    if use_double && use_float {
        eprintln!("Error: Cannot specify both --f64 and --f32");
        std::process::exit(1);
    }

    let dim = 2usize.pow(n as u32);

    if use_double {
        println!("Using f64 precision for matrix multiplication.");
        let (ave_gflops, max_gflops, ave_time, min_time) = execute_double(dim, loop_num);
        println!(
            "Average Gflops: {:.3}, Max Gflops: {:.3}",
            ave_gflops, max_gflops
        );
        println!("Average Time: {:.6}s, Min Time: {:.6}s", ave_time, min_time);
    } else {
        println!("Using f32 precision for matrix multiplication.");
        let (ave_gflops, max_gflops, ave_time, min_time) = execute_float(dim, loop_num);
        println!(
            "Average Gflops: {:.3}, Max Gflops: {:.3}",
            ave_gflops, max_gflops
        );
        println!("Average Time: {:.6}s, Min Time: {:.6}s", ave_time, min_time);
    }
}
