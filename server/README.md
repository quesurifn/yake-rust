# YAKE Rust Server

A fast, efficient keyword extraction HTTP server built with Rust, using the YAKE (Yet Another Keyword Extractor) algorithm.

## Features

- High-performance keyword extraction using Rust implementation of YAKE
- RESTful API with JSON request/response format
- Support for 34 languages
- Configurable n-gram size and result count
- Lightweight Docker container based on Alpine Linux
- Cross-platform support (x86_64 and ARM64/aarch64)

## Getting Started

### Using Docker

The easiest way to run the YAKE server is using Docker:

```bash

# Run the container
docker run -p 8080:8080 yake-server:<arch>
```

Environment variables:
- `SRV_PORT`: Server port (default: 8080)
- `RUST_LOG`: Log level (default: info)
- `WORKERS`: Number of worker threads (default: 4)

### Building from Source

To build the Docker image:

```bash
# For x86_64
docker build --build-arg TARGETARCH=x86_64 -t yake-server:x86_64 .

# For ARM64/Apple Silicon
docker build --build-arg TARGETARCH=aarch64 -t yake-server:aarch64 .

# Note: This Dockerfile structure requires distinct runtime stages
# for each architecture due to Docker's limitations with variable 
# interpolation in the --from argument
```

For multi-platform builds (requires Docker BuildX):

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t yake-server:latest .
```

## API Usage

### Base URL

`http://localhost:8080/`

### Endpoints

#### GET `/`

Returns a simple greeting.

**Response**:
```
Hello, Yake!
```

#### POST `/keywords`

Extract keywords from text.

**Request Body**:
```json
{
  "body": "Your text content goes here. This is the text from which keywords will be extracted.",
  "ngrams": 3,
  "language": "en",
  "num_results": 10
}
```

Parameters:
- `body`: Text content to analyze
- `ngrams`: Maximum size of word phrases to extract (default: 3)
- `language`: ISO language code (e.g., "en" for English)
- `num_results`: Number of keywords to return

**Response Body**:
```json
[
  {
    "raw": "keyword phrase",
    "keyword": "keyword phrase",
    "score": 0.023
  },
  ...
]
```

## Supported Languages

The server supports the following languages:

| Code | Language        | Code | Language        | Code | Language        |
|------|-----------------|------|-----------------|------|-----------------|
| ar   | Arabic          | bg   | Bulgarian       | br   | Brazilian       |
| cz   | Czech           | da   | Danish          | de   | German          |
| el   | Greek           | en   | English         | es   | Spanish         |
| et   | Estonian        | fa   | Persian         | fi   | Finnish         |
| fr   | French          | hi   | Hindi           | hr   | Croatian        |
| hu   | Hungarian       | hy   | Armenian        | id   | Indonesian      |
| it   | Italian         | ja   | Japanese        | lt   | Lithuanian      |
| lv   | Latvian         | nl   | Dutch           | no   | Norwegian       |
| pl   | Polish          | pt   | Portuguese      | ro   | Romanian        |
| ru   | Russian         | sk   | Slovak          | sl   | Slovenian       |
| sv   | Swedish         | tr   | Turkish         | uk   | Ukrainian       |
| zh   | Chinese         |      |                 |      |                 |

## License

This project is licensed under the MIT License.

## Acknowledgments

- [YAKE Rust](https://github.com/quesurifn/yake-rust) - The Rust implementation of the YAKE algorithm
- [Axum](https://github.com/tokio-rs/axum) - A web application framework for Rust