# Dacon Hidden Letters Classification

## Progress Status

| Task | Status | Description |
|------|--------|-------------|
| Data Loading | Done | CSV 파일 로드 및 Google Drive 마운트 |
| Data Preprocessing | Done | 이미지 reshape (28x28x1), 정규화 (0-1) |
| Data Augmentation | Done | Height/Width shift 적용 |
| Model Architecture | Done | CNN 모델 (Conv2D + BatchNorm + Dropout) |
| K-Fold Cross Validation | Done | 40-Fold Stratified K-Fold |
| GPU Optimization | Done | Mixed Precision, XLA, tf.data API |
| Training | Pending | 사용자 실행 대기 중 |
| Submission | Pending | 학습 완료 후 제출 파일 생성 |

---

## Files

| File | Description |
|------|-------------|
| `main.ipynb` | 원본 노트북 |
| `main_optimized.ipynb` | A100 GPU 최적화 버전 |
| `train.csv` | 학습 데이터 |
| `test.csv` | 테스트 데이터 |
| `submission.csv` | 제출 템플릿 |

---

## A100 GPU Optimization Details

### Original vs Optimized

| 항목 | Original | Optimized | 예상 속도 향상 |
|------|----------|-----------|----------------|
| 연산 정밀도 | FP32 | Mixed FP16 | 2-3x |
| 배치 크기 | 8 | 64 | 3-5x |
| 데이터 파이프라인 | ImageDataGenerator | tf.data + prefetch | 1.5-2x |
| XLA 컴파일 | 비활성화 | 활성화 | 1.2-1.5x |
| **총 예상 속도 향상** | - | - | **5-10x** |

---

## Code Explanation

### 1. Environment Setup (GPU Optimization)

```python
# GPU Memory Growth - OOM 방지
tf.config.experimental.set_memory_growth(gpu, True)

# XLA Compiler 활성화 - GPU 연산 최적화
tf.config.optimizer.set_jit(True)

# Mixed Precision (FP16) - A100 Tensor Core 활용
mixed_precision.set_global_policy('mixed_float16')
```

**원리:**
- **Memory Growth**: GPU 메모리를 필요할 때만 할당하여 OOM(Out of Memory) 에러 방지
- **XLA (Accelerated Linear Algebra)**: TensorFlow 연산을 컴파일하여 GPU 커널 융합, 불필요한 메모리 연산 제거
- **Mixed Precision**: FP16으로 연산하고 FP32로 가중치 저장. A100의 Tensor Core는 FP16 연산에서 최대 성능 발휘

---

### 2. Data Loading

```python
train = pd.read_csv(base_path + 'train.csv')
test = pd.read_csv(base_path + 'test.csv')
```

**데이터 구조:**
- `train.csv`: 2048개 샘플 x 787 columns (id, digit, letter, pixel 0-783)
- `test.csv`: 20480개 샘플 x 786 columns (id, letter, pixel 0-783)
- 이미지 크기: 28x28 픽셀 (784개 픽셀 값)
- 레이블: 0-9 숫자 (10개 클래스)

---

### 3. Data Preprocessing

```python
# 이미지 형태로 reshape
train_features = train_features.reshape(-1, 28, 28, 1)

# 정규화 (0-255 -> 0-1)
train_features = train_features.astype(np.float32) / 255.0
```

**원리:**
- **Reshape**: (N, 784) -> (N, 28, 28, 1) CNN 입력 형태로 변환
- **Normalization**: 픽셀 값을 0-1 범위로 정규화하여 학습 안정성 향상
- **float32**: Mixed Precision에서 자동으로 FP16으로 캐스팅됨

---

### 4. Optimized Data Pipeline (tf.data API)

```python
@tf.function
def augment_image(image, label):
    # Pad -> Random crop으로 shift 효과 구현
    image = tf.image.pad_to_bounding_box(image, 1, 1, 30, 30)
    image = tf.image.random_crop(image, size=[28, 28, 1])
    return image, label

def create_train_dataset(x, y, batch_size, augment=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset
```

**원리:**
- **tf.data API**: ImageDataGenerator보다 효율적인 데이터 파이프라인
- **@tf.function**: 함수를 TensorFlow 그래프로 컴파일하여 GPU에서 실행
- **prefetch(AUTOTUNE)**: 데이터 로딩과 모델 연산을 병렬화
- **num_parallel_calls=AUTOTUNE**: 멀티스레드로 데이터 전처리

**Original (ImageDataGenerator) vs Optimized (tf.data):**
```
Original:
CPU [데이터 로드] -> [전처리] -> GPU [학습] -> CPU [데이터 로드] -> ...

Optimized:
CPU [데이터 로드 + 전처리] -----> [다음 배치 준비]
GPU [학습] ----------------------> [학습] -------> ...
         ↑ prefetch로 대기 시간 제거
```

---

### 5. Model Architecture

```python
model = Sequential([
    # Block 1: Feature Extraction (Low-level)
    Conv2D(16, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    # Block 2: Feature Extraction (Mid-level)
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (5,5), activation='relu', padding='same'),  # x3
    BatchNormalization(),
    MaxPooling2D((3,3)),
    Dropout(0.3),

    # Block 3: Feature Extraction (High-level)
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (5,5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((3,3)),
    Dropout(0.3),

    # Classification Head
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax', dtype='float32')  # FP32로 출력
])
```

**원리:**
- **Conv2D**: Convolutional layer로 이미지의 지역적 특징 추출
- **BatchNormalization**: 내부 공변량 변화 감소, 학습 안정화
- **Dropout(0.3)**: 30% 뉴런을 무작위로 비활성화하여 과적합 방지
- **MaxPooling2D**: 공간 해상도 축소, 위치 불변성 확보
- **출력층 dtype='float32'**: Mixed Precision에서 softmax의 수치 안정성 보장

**Feature Map 크기 변화:**
```
Input: 28x28x1
  ↓ Conv2D + BatchNorm + Dropout
28x28x16
  ↓ Conv2D x4 + BatchNorm
28x28x32
  ↓ MaxPooling(3,3)
9x9x32
  ↓ Conv2D x2 + BatchNorm
9x9x64
  ↓ MaxPooling(3,3)
3x3x64
  ↓ Flatten
576
  ↓ Dense
128 -> 64 -> 10 (output)
```

---

### 6. K-Fold Cross Validation

```python
skf = StratifiedKFold(n_splits=40, random_state=42, shuffle=True)

for train_index, valid_index in skf.split(train_features, train_labels):
    # 각 Fold마다 독립적으로 모델 학습
    model = build_model()
    model.fit(train_dataset, validation_data=valid_dataset, ...)

    # 예측 결과 앙상블 (평균)
    result += model.predict(test_dataset) / N_SPLITS
```

**원리:**
- **Stratified K-Fold**: 각 Fold에서 클래스 비율 유지
- **40-Fold**: 데이터가 작아서 (2048개) 많은 Fold로 안정적인 검증
- **앙상블**: 40개 모델의 예측을 평균하여 분산 감소, 일반화 성능 향상

```
Fold 1: Train on 1997 samples, Validate on 51 samples
Fold 2: Train on 1997 samples, Validate on 51 samples
...
Fold 40: Train on 1997 samples, Validate on 51 samples

Final Prediction = (Pred_1 + Pred_2 + ... + Pred_40) / 40
```

---

### 7. Callbacks

```python
# Learning Rate Scheduler
ReduceLROnPlateau(patience=100, factor=0.5)

# Early Stopping
EarlyStopping(patience=160, restore_best_weights=True)

# Model Checkpoint
ModelCheckpoint('best_model.h5', save_best_only=True)
```

**원리:**
- **ReduceLROnPlateau**: 검증 손실이 100 에폭 동안 개선되지 않으면 LR을 0.5배로 감소
- **EarlyStopping**: 160 에폭 동안 개선 없으면 학습 중단
- **ModelCheckpoint**: 가장 좋은 검증 손실의 가중치 저장

```
Epoch    Loss    LR
1        0.8     0.002
...
100      0.3     0.002  (plateau detected)
101      0.3     0.001  (LR reduced)
...
200      0.25    0.001  (plateau detected)
201      0.25    0.0005 (LR reduced)
...
```

---

### 8. Mixed Precision Training (A100 Optimization)

```python
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

**A100 Tensor Core 활용:**

```
기존 (FP32):
┌─────────────────────────────────┐
│  FP32 Matrix A  x  FP32 Matrix B │
│       (32-bit)       (32-bit)     │
│              ↓                    │
│       FP32 Result                 │
│       (32-bit)                    │
└─────────────────────────────────┘
연산량: 19.5 TFLOPS (A100)

Mixed Precision (FP16):
┌─────────────────────────────────┐
│  FP16 Matrix A  x  FP16 Matrix B │
│       (16-bit)       (16-bit)     │
│              ↓                    │
│       FP32 Accumulator            │
│       (32-bit, 정밀도 유지)        │
└─────────────────────────────────┘
연산량: 312 TFLOPS (A100 Tensor Core)

→ 약 16배 더 빠른 행렬 연산!
```

**주의사항:**
- Loss Scaling: 작은 gradient가 FP16에서 언더플로우되지 않도록 자동 스케일링
- 출력층은 FP32 유지: softmax의 수치 안정성

---

### 9. Prediction & Submission

```python
# 앙상블 예측 결과에서 가장 높은 확률의 클래스 선택
sub['digit'] = result.argmax(axis=1)
sub.to_csv('submission_optimized_A100.csv', index=False)
```

**원리:**
- `result`: (20480, 10) 형태의 확률 분포
- `argmax(axis=1)`: 각 샘플에서 가장 높은 확률의 클래스 인덱스 선택

---

## Performance Tips

1. **Colab Pro+ 사용**: A100 GPU 할당 확률 증가
2. **런타임 유형 설정**: 런타임 > 런타임 유형 변경 > GPU (A100)
3. **세션 유지**: 학습 중 브라우저 탭 활성 상태 유지
4. **체크포인트 저장**: Google Drive에 모델 저장으로 중단 대비

---

## Expected Training Time

| GPU | Original (Batch=8) | Optimized (Batch=64) |
|-----|-------------------|----------------------|
| T4  | ~8-10 hours       | ~2-3 hours           |
| V100 | ~4-5 hours       | ~1-1.5 hours         |
| A100 | ~2-3 hours       | ~20-40 minutes       |

*40-Fold 전체 학습 기준, Early Stopping에 따라 변동*
