This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## AI Semantic Segmentation Backend

This project integrates a Deep Learning module (U-Net) to perform semantic segmentation on Martian terrain images. The backend logic is located in the `backend/` directory.

### Prerequisites

* Python 3.9+
* pip (Python Package Installer)

### Backend Setup

Navigate to the backend directory and install the required dependencies:

```
cd backend
pip install tensorflow opencv-python matplotlib kagglehub scikit-learn python-dotenv
```

### Running the AI Pipeline

The backend includes modular scripts for data management and model testing. Execute them in the following order:

1. Data Ingestion & Pre-processing Verification

Downloads the AI4Mars dataset (if not present) and runs a visual diagnostic to verify the rover masking pipeline.


```
python data_setup.py
```
Output: A success message and a popup window comparing the "Rover Masked Input" vs. "Ground Truth".

2. Architecture Integration Test

Runs a lightweight training simulation (Debug Mode) using a small subset of images. This validates the U-Net architecture layers and the Data Augmentation pipeline locally before deploying to the cloud.

```
python prueba_local.py
```
Output: Progress bars for model construction and a final message confirming: ✅ PRUEBA EXITOSA: Pipeline de datos y arquitectura validados.

### Technical Architecture (White-Box Design)

* Model: Custom U-Net Implementation (Encoder-Decoder with Skip Connections).
* Framework: TensorFlow/Keras (Functional API).
* Pre-processing: OpenCV-based artifact removal (Rover Masking) and Normalization.
* Data Augmentation: Real-time geometric transformations (Flip, Rotation) and photometric adjustments (Brightness) to prevent overfitting.