# System Architecture

## Automated Deepfake Detection System

**Project abbreviation:** ADDFS  
**Document version:** 1.0

---

## 1. Architecture Overview

ADDFS uses a modular, layered architecture so that video processing, machine-learning inference, reporting, and the user interface can be developed and tested independently.

The first release will run locally and process manually uploaded video files without relying on cloud-based inference.

---

## 2. High-Level Processing Flow

```text
User
  |
  v
Web Interface
  |
  v
Upload Validation
  |
  v
Video Ingestion
  |
  v
Frame Extraction
  |
  v
Face Detection and Cropping
  |
  v
Image Preprocessing
  |
  v
Deepfake Classification Model
  |
  v
Prediction Aggregation
  |
  v
Explainability and Evidence Selection
  |
  v
Forensic Result and Report