
#[allow(unused_imports)]
use hello_rust::engine::{ai_engine, random_engine};
use hello_rust::game::fortress::Game;

use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark_rand(c: &mut Criterion) {
    c.bench_function("random games", |b| b.iter(|| main_rand()));
}

pub fn criterion_benchmark_ai_rand(c: &mut Criterion) {
    c.bench_function("ai vs random games", |b| b.iter(|| main_ai_rand()));
}

pub fn criterion_benchmark_ai(c: &mut Criterion) {
    c.bench_function("ai vs ai games", |b| b.iter(|| main_ai()));
}

pub fn criterion_benchmark_train(c: &mut Criterion) {
    c.bench_function("train only", |b| b.iter(|| main_train()));
}