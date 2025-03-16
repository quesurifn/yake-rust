use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{extract, routing::get, Json, Router};
use config::Config;
use log::info;
use serde::{Deserialize, Serialize};
use yake_rust::ResultItem;
use yake_rust::{get_n_best, Config as YakeConfig, StopWords};

#[derive(Deserialize, Debug)]
pub struct YakeRequest {
    body: String,
    ngrams: usize,
    language: String,
    num_results: usize,
}

#[derive(Serialize, Debug)]
pub struct YakeResponse<T> {
    success: bool,
    message: String,
    data: T,
}

impl YakeResponse<ResultItem> {
    pub fn new(data: ResultItem) -> Self {
        Self { success: true, message: String::from("success"), data }
    }
}

impl YakeResponse<String> {
    pub fn error(message: &str) -> Self {
        Self { success: false, message: String::from(message), data: String::new() }
    }
}

#[tokio::main]
async fn main() {
    let settings = Config::builder().add_source(config::Environment::with_prefix("SRV")).build().unwrap();

    let port = match settings.get_int("port") {
        Ok(int) => int,
        Err(_) => panic!("Port is not configured"),
    };

    let app = Router::new().route("/", get(root)).route("/keywords", post(keywords));
    let host_str = format!("0.0.0.0:{port}", port = port);

    let listener = tokio::net::TcpListener::bind(host_str.clone()).await.unwrap();
    info!("Yake Srv is now listening on {}!", host_str.clone());
    axum::serve(listener, app).await.unwrap();
}

async fn root() -> &'static str {
    "Hello, Yake!"
}

async fn keywords(extract::Json(payload): extract::Json<YakeRequest>) -> Response {
    info!("Serving request: {:?}", payload);
    let supported_languages = vec![
        "ar", "bg", "br", "cz", "da", "de", "el", "en", "es", "et", "fa", "fi", "fr", "hi", "hr", "hu", "hy", "id",
        "it", "ja", "lt", "lv", "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "zh",
    ];

    if !supported_languages.contains(&&payload.language[..]) {
        return Json(YakeResponse::<String>::error("Language not supported")).into_response();
    }

    let mut yake_cfg = YakeConfig::default();
    if payload.ngrams != yake_cfg.ngrams {
        yake_cfg.ngrams = payload.ngrams;
    }
    let ignored = StopWords::predefined(payload.language).unwrap();

    let keywords = get_n_best(payload.num_results, &*payload.body, &ignored, yake_cfg);

    Json(keywords).into_response()
}
