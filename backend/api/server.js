import dotenv from "dotenv";
import express from "express";
import cors from "cors";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";

import pkg from "../generated/prisma/client.js";
const { PrismaClient } = pkg;

import { PrismaNeon } from "@prisma/adapter-neon";

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json());

// ============================
// Prisma + Neon Adapter Setup
// ============================
const connectionString = process.env.DATABASE_URL;

const adapter = new PrismaNeon({
  connectionString,
});

const prisma = new PrismaClient({
  adapter,
});

// ============================
// Multer setup (memory)
// ============================
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 4 * 1024 * 1024 },
});

// ============================
// Health Check
// ============================
app.get("/api", async (req, res) => {
  try {
    await prisma.$queryRaw`SELECT 1`;
    res.status(200).json({
      message: "Fly Counting Backend Running 🚀",
      db: "connected",
    });
  } catch (error) {
    res.status(500).json({
      message: "Backend running but DB error",
      error: error.message,
    });
  }
});

// ============================
// Upload & Process Image
// ============================
app.post("/api/upload", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "Image file is required" });

    if (!req.body.location) return res.status(400).json({ error: "Location is required" });

    const { location } = req.body;

    const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:8000/predict";

    const formData = new FormData();
    formData.append("file", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const mlResponse = await axios.post(ML_SERVICE_URL, formData, {
      headers: formData.getHeaders(),
    });

    const flyCount = mlResponse.data.total || 0;

    const inspection = await prisma.inspection.create({
      data: {
        location,
        flyCount,
        imageUrl: req.file.originalname,
      },
    });

    res.status(200).json({
      message: "Upload & detection success",
      detection: mlResponse.data,
      savedData: inspection,
    });
  } catch (error) {
    console.error("UPLOAD ERROR:", error);

    res.status(500).json({
      error: "Upload failed",
      detail: error.message,
    });
  }
});

// ============================
// Get inspection history
// ============================
app.get("/api/inspection", async (req, res) => {
  try {
    const inspections = await prisma.inspection.findMany({
      orderBy: { createdAt: "desc" },
    });

    res.status(200).json(inspections);
  } catch (error) {
    console.error("FETCH ERROR:", error);

    res.status(500).json({
      error: "Failed to fetch inspections",
    });
  }
});

// ============================
// Start Server
// ============================
const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});

export default app;
