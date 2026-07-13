# Software Requirements Specification

## Automated Deepfake Detection System

**Project abbreviation:** ADDFS  
**Document version:** 1.0  
**Status:** Initial development specification

---

## 1. Introduction

### 1.1 Purpose

This document defines the software requirements for the Automated Deepfake Detection System, abbreviated as ADDFS.

ADDFS is a defensive cybersecurity and digital-forensics system that analyzes uploaded video files and estimates whether visible facial content is authentic or artificially manipulated.

The system is intended to support researchers, journalists, financial institutions, security personnel, and other authorized users who need an accessible method of examining potentially manipulated video content.

### 1.2 Project Objective

The main objective is to design and develop a lightweight, deep-learning-driven system capable of detecting facial deepfakes in compressed, real-world video.

The system will focus on:

- facial structural irregularities;
- visual inconsistencies between frames;
- unnatural facial movement patterns;
- face-swap and face-replacement manipulation;
- degradation caused by video compression and resizing.

### 1.3 Intended Users

The intended users include:

- cybersecurity researchers;
- digital-forensics investigators;
- journalists and fact-checkers;
- financial-fraud investigators;
- university researchers and students;
- authorized institutional security personnel.

### 1.4 Product Scope

The first version of ADDFS will:

1. Accept a video uploaded manually by a user.
2. Validate the file type, size, resolution, duration, and readability.
3. Extract representative frames from the video.
4. Detect and crop visible facial regions.
5. Analyze facial frames using a trained deep-learning model.
6. Combine frame-level predictions into a video-level result.
7. Calculate a deepfake probability score.
8. Present the result through a local web interface.
9. Display supporting evidence such as suspicious frames or heatmaps.
10. Record processing information for testing and evaluation.

The first version will not:

- analyze audio or detect cloned voices;
- automatically download videos from social-media platforms;
- monitor live CCTV or livestream feeds;
- provide a mobile application;
- guarantee correct detection for every deepfake technique;
- make legal conclusions about the authenticity of evidence.