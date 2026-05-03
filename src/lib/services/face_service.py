from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import onnxruntime
from lib.schemas import EmbeddingRecord, FaceDetection, PredictResult, AlignedFace
from lib.storage.base import EmbeddingStoreProtocol
import os 
import logging
from facenet_pytorch import MTCNN
from PIL import Image

logger = logging.getLogger(__name__)


class FaceService:
    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        face_size: int,
        model_path: Path,
        output_path: Path = Path("output"),
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.face_size = face_size
        self.output_path = output_path


        self._fa = MTCNN(
            image_size=face_size,
            keep_all=True,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            margin=20,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),)
        

        self.model = self._load_model(model_path)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)
        self.model.eval()

        os.makedirs(self.output_path, exist_ok=True)
        logger.info(f"FaceService inicializado | dispositivo: {self._device}")

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    @staticmethod
    def _kps_to_keypoints_dict(kps: np.ndarray | None) -> dict[str, list[int]]:
        if kps is None or len(kps) == 0:
            return {}
        return {
            f"k{i}": [int(round(float(kps[i, 0]))), int(round(float(kps[i, 1])))]
            for i in range(len(kps))
        }


    def _load_model(self, model_path: Path) -> any:
        mp = Path(model_path)
        if not mp.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        suf = mp.suffix.lower()
        if suf == ".pth":
            return torch.load(mp, map_location="cpu", weights_only=False)
        if suf == ".onnx":
            return onnxruntime.InferenceSession(str(mp))
        raise ValueError(f"Unsupported model format (expected .pth or .onnx): {model_path}")

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(source_path)
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (InsightFace / OpenCV convention)
        return image

    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detecta todas las caras presentes en una imagen  usando MTCNN.
        Recibe imagen BGR, devuelve lista de (x1, y1, x2, y2).
        """

        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Convierte la imagen de BGR (OpenCV) a RGB (PIL)  MTCNN 
        boxes, probs = self._fa.detect(img_pil, landmarks=False) # detecta cara y devuelve los 4 puntos y probs

        if boxes is None:
            return []

        resultado = []
        for box, prob in zip(boxes, probs):
            if prob < 0.9:          # filtra detecciones con probabilidad baja
                continue
            x1, y1, x2, y2 = self._clip_xyxy(
                int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                image.shape[0], image.shape[1]
            )                                            #Ajusta las coordenadas con _clip_xyxy para que no salgan fuera de la imagen
            resultado.append((x1, y1, x2, y2))

        return resultado


    def align_face(
        self, image: np.ndarray, box: tuple[int, int, int, int]) -> AlignedFace:
        """
        Alinea la cara más cercana al box dado usando MTCNN.
        Recibe imagen BGR, devuelve AlignedFace con imagen BGR normalizada.
        """
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convierte la imagen de BGR (OpenCV) a RGB (PIL)
        boxes, probs, landmarks = self._fa.detect(img_pil, landmarks=True) # 5 landmarks
        faces_tensor = self._fa(img_pil)   # (N, 3, face_size, face_size) en [-1, 1]

        x1, y1, x2, y2 = box

        # Si hay más de una cara, buscar la más cercana al box dado
        best_idx = 0
        if boxes is not None and len(boxes) > 1:
            min_dist = float("inf")
            for i, b in enumerate(boxes):
                dist = abs(int(b[0]) - x1) + abs(int(b[1]) - y1)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i

        kps = landmarks[best_idx] if landmarks is not None else None

        # Si MTCNN no pudo generar el tensor alineado hacer recorte simple
        if faces_tensor is None:
            x1c, y1c, x2c, y2c = self._clip_xyxy(
               x1, y1, x2, y2, image.shape[0], image.shape[1]
            )
            crop = cv2.resize(
                image[y1c:y2c, x1c:x2c],
                (self.face_size, self.face_size)
            )
            if kps is not None:
                kps = np.array([[pt[0] - x1, pt[1] - y1] for pt in kps]) #Convertir landmarks de coordenadas absolutas a relativas al crop
            bbox = np.array(boxes[best_idx]) if boxes is not None else np.array(box)
            return AlignedFace(bbox=bbox, keypoints=kps, image=face_bgr, embedding=None)

        # Convertir tensor [-1,1] → BGR uint8
        t = faces_tensor[best_idx] if faces_tensor.ndim == 4 else faces_tensor
        face_np = ((t.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)

        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR) # Convertir RGB → BGR para mantener la convención de OpenCV en el resto del sistema

        bbox = np.array(boxes[best_idx]) if boxes is not None else np.array(box)
        if kps is not None:
            kps = np.array([[pt[0] - x1, pt[1] - y1] for pt in kps]) #Convertir landmarks de coordenadas absolutas a relativas al crop

        bbox = np.array(boxes[best_idx]) if boxes is not None else np.array(box)
        return AlignedFace(bbox=bbox, keypoints=kps, image=face_bgr, embedding=None)


    def extract_embedding_from_face(self, face: AlignedFace) -> list[float]:
        """
        Extrae embedding de 512 dimensiones usando InceptionResnetV1.
        Recibe AlignedFace con imagen BGR uint8, devuelve lista de 512 floats.
        """
        
        face_rgb = cv2.cvtColor(face.image, cv2.COLOR_BGR2RGB) # Convertir BGR uint8 → tensor RGB normalizado [-1, 1]
        tensor = torch.tensor(face_rgb, dtype=torch.float32).permute(2, 0, 1)
        tensor = (tensor - 127.5) / 128.0  # normalización estándar de FaceNet
        tensor = tensor.unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self.model(tensor)

        emb = embedding.squeeze().cpu().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.tolist()

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    def identify(self, query_embedding: list[float]) -> tuple[str, float]:
        records = self.store.all()
        if not records:
            return "unknown", 0.0

        best_label = "unknown"
        best_score = -1.0
        for record in records:
            score = self.similarity(query_embedding, record.embedding)
            if score > best_score:
                best_score = score
                best_label = record.etiqueta

        if best_score < self.similarity_threshold:
            return "unknown", max(best_score, 0.0)
        return best_label, best_score

    def register_identity(
        self, identity: str, image_path: str, metadata: dict[str, object]
    ) -> EmbeddingRecord:
        image = self._load_image(image_path)
        faces = self.detect_faces(image)

        if len(faces) != 1:
            raise ValueError("Exactly one face must be detected for identity registration.")
        
        logger.info(f"Face detected: {faces[0]}")

        box = faces[0]
        aligned = self.align_face(image, box)
        embedding = self.extract_embedding_from_face(aligned)

        img_id = str(uuid4())
        img_output_path = self.output_path / f"img_{img_id}.jpg"
        
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=str(img_output_path),
            etiqueta=identity,
            metadata=metadata,
        )
        self.store.append(record)

        cv2.imwrite(str(img_output_path), aligned.image)
        logger.info(f"Identity registered: {identity} with image: {image_path}")
        return record

    def predict(self, source_path: str, output_path: Path) -> str:
        image = self._load_image(source_path)
        faces = self.detect_faces(image)
        detections: list[FaceDetection] = []
        for (x1, y1, x2, y2) in faces:
            aligned = self.align_face(image, (x1, y1, x2, y2))
            embedding = self.extract_embedding_from_face(aligned)
            label, score = self.identify(embedding)
            kps = getattr(aligned, "keypoints", None)
            kps_arr = np.asarray(kps) if kps is not None else None
            detections.append(
                FaceDetection(
                    bbox=[x1, y1, x2, y2],
                    keypoints=self._kps_to_keypoints_dict(kps_arr),
                    label=label,
                    score=round(float(score), 4),
                )
            )

        detected_people = sorted({item.label for item in detections if item.label != "unknown"})
        result_payload = PredictResult(
            source_path=source_path,
            detections=detections,
            detected_people=detected_people,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(result_payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
