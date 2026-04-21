# Kobun (古文) — Classical Japanese Document Recognition

> Read classical Japanese kuzushiji manuscripts with deep learning.
> 古文書をAIで解読する。ディープラーニングによる崩し字認識プロジェクト。

**Status**: Early Development (Phase 0 — Setup)

Full-stack ML system for recognizing and translating classical Japanese
(kuzushiji) documents. Built with PyTorch, YOLO, and ONNX, served through a
trilingual web interface (Japanese / English / Indonesian).

---

## Demo

_Coming soon._ Public HuggingFace Space + Vercel web app will be linked here
once Phase 3 (ML Server) and Phase 4 (Frontend) are complete.

## Architecture

Monorepo with three deployable surfaces: Next.js frontend (Vercel), FastAPI
proxy backend (Vercel Serverless), and a Gradio ML server (HuggingFace
Spaces). Full diagram and tech stack details will be added in the final
technical report (Phase 5).

## Installation

Reproducible setup via `make setup`, `make data`, `make train-all`, and
`make dev-frontend` will be documented in Phase 5. Target: clone-to-demo in
under two hours on a fresh environment.

## Citation

If you use this work, please cite the underlying Kuzushiji dataset:

> Clanuwat, T., Bober-Irizar, M., Kitamoto, A., Lamb, A., Yamamoto, K., & Ha, D.
> (2018). *Deep Learning for Classical Japanese Literature*. arXiv:1812.01718.

Citation metadata for this project will be provided in `CITATION.cff`.

## License

Released under the [MIT License](./LICENSE). Dataset usage follows the
CC BY-SA 4.0 terms of the original Kuzushiji corpus (Clanuwat et al. 2018).
