# 디퓨전 모델의 발전 과정: DDPM부터 Flow Matching까지

디퓨전 모델은 현재 이미지 생성, 비디오 생성, 오디오 생성, 3D 생성, 로봇 행동 생성 등에서 널리 사용된다. 하지만 처음부터 지금과 같은 형태였던 것은 아니다.

발전 과정을 아주 크게 요약하면 다음과 같다.

1. 데이터에 노이즈를 조금씩 넣고 그 과정을 거꾸로 학습하는 아이디어가 등장했다.
2. Score Matching이 확률분포의 밀도 자체가 아니라 밀도가 증가하는 방향을 학습하는 관점을 제공했다.
3. DDPM이 단순한 노이즈 예측 손실과 U-Net을 결합해 높은 생성 품질을 보였다.
4. DDIM, DPM-Solver, Distillation 등이 느린 샘플링을 개선했다.
5. Latent Diffusion이 픽셀 대신 압축된 latent 공간에서 확산을 수행했다.
6. DiT가 U-Net 대신 Transformer를 디퓨전 모델의 backbone으로 사용했다.
7. Flow Matching과 Rectified Flow가 노이즈에서 데이터로 이동하는 속도장을 직접 학습하는 방향을 발전시켰다.

이 글에서는 이 흐름을 수식과 함께 차근차근 설명한다.

---

# 0. 먼저 사용할 기호

수식에서 같은 기호를 서로 다른 의미로 사용하면 혼란스럽기 때문에, 이 글에서는 다음과 같이 통일한다.

| 기호 | 의미 |
|---|---|
| $\mathbf{x}_0$ | 원본 데이터 또는 clean data |
| $\mathbf{x}_t$ | 이산 디퓨전 단계 $t$에서의 noisy data |
| $t$ | 이산 디퓨전 단계, $t\in\{1,\ldots,T\}$ |
| $T$ | 전체 디퓨전 단계 수 |
| $\beta_t$ | $t$번째 정방향 단계에서 추가하는 노이즈의 분산 |
| $\alpha_t$ | $1-\beta_t$ |
| $\bar{\alpha}_t$ | $\prod_{s=1}^{t}\alpha_s$, 누적 신호 보존율 |
| $\boldsymbol{\varepsilon}$ | 학습용으로 샘플링한 표준 가우시안 노이즈 |
| $\boldsymbol{\varepsilon}_\theta$ | 신경망이 예측한 노이즈 |
| $\boldsymbol{\xi}_t$ | 역방향 샘플링 중 새로 넣는 표준 가우시안 노이즈 |
| $\sigma_t$ | 역방향 샘플링 노이즈의 표준편차 |
| $\mathbf{s}_\theta$ | 신경망이 근사한 score |
| $\mathbf{c}$ | 클래스, 텍스트, 관측 등 조건 정보 |
| $\tau$ | 연속시간, 보통 $\tau\in[0,1]$ |
| $\mathbf{r}_\tau$ | Score-SDE에서 연속시간 상태 |
| $\mathbf{y}_\tau$ | Flow Matching에서 연속시간 상태 |
| $\mathbf{u}_\theta$ | Flow Matching의 velocity field |
| $d$ | 데이터 벡터의 차원 |

여기서 $\mathbf{x}$는 이미지에만 한정되지 않는다. 이미지라면 모든 픽셀을 모은 고차원 벡터이고, 로봇에서는 행동 시퀀스일 수도 있다.

---

# 1. 생성 모델이 하려는 일

실제 데이터가 어떤 확률분포

```math
\mathbf{x}_0\sim p_{\mathrm{data}}(\mathbf{x})
```

에서 나왔다고 하자.

생성 모델의 목표는 실제 데이터 분포 $p_{\mathrm{data}}$와 비슷한 모델 분포 $p_\theta$를 학습하는 것이다.

```math
p_\theta(\mathbf{x})
\approx
p_{\mathrm{data}}(\mathbf{x})
```

학습이 끝나면 모델에서 새로운 샘플을 뽑는다.

```math
\hat{\mathbf{x}}_0
\sim
p_\theta(\mathbf{x})
```

문제는 실제 데이터 분포가 매우 복잡하다는 것이다.

예를 들어 이미지 한 장이 $512\times512$ RGB 이미지라면 차원은

```math
d=512\times512\times3
```

이다. 이 고차원 공간에서 자연스러운 이미지가 존재하는 영역은 매우 복잡한 형태를 가진다.

GAN은 생성기와 판별기의 경쟁으로 이 분포를 학습했고, VAE는 latent variable과 변분추론을 이용했다. 디퓨전 모델은 다른 전략을 사용한다.

> 복잡한 데이터 분포를 한 번에 직접 만들지 말고, 데이터를 단순한 노이즈 분포로 조금씩 망가뜨린 뒤 그 과정을 거꾸로 학습하자.

---

# 2. 디퓨전의 가장 기본적인 직관

원본 이미지에 아주 작은 가우시안 노이즈를 반복해서 더한다고 생각해 보자.

```math
\mathbf{x}_0
\rightarrow
\mathbf{x}_1
\rightarrow
\mathbf{x}_2
\rightarrow
\cdots
\rightarrow
\mathbf{x}_T
```

초기에는 원본 구조가 잘 보이지만, 시간이 지날수록 점점 흐려지고 마지막에는 거의 순수한 가우시안 노이즈가 된다.

```math
\mathbf{x}_T
\approx
\mathcal{N}(\mathbf{0},\mathbf{I}_d)
```

이 과정을 **정방향 과정**, forward process 또는 diffusion process라고 한다.

그다음 반대로

```math
\mathbf{x}_T
\rightarrow
\mathbf{x}_{T-1}
\rightarrow
\cdots
\rightarrow
\mathbf{x}_1
\rightarrow
\mathbf{x}_0
```

노이즈를 조금씩 제거하는 과정을 학습한다. 이를 **역방향 과정**, reverse process 또는 denoising process라고 한다.

생성할 때는 실제 데이터가 필요하지 않다. 먼저 순수한 노이즈를 만든 다음, 학습한 역방향 과정을 반복한다.

```math
\text{random noise}
\rightarrow
\text{structured noise}
\rightarrow
\text{coarse data}
\rightarrow
\text{clean data}
```

---

# 3. 2015년: Diffusion Probabilistic Model

디퓨전 생성 모델의 초기 형태는 2015년 논문 [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)에서 제안되었다.

핵심 아이디어는 비평형 열역학의 확산 과정에서 영감을 받았다.

- 정방향 과정에서는 복잡한 데이터 분포를 단순한 가우시안 분포로 천천히 변환한다.
- 역방향 과정에서는 가우시안 노이즈를 데이터 분포로 되돌리는 전이를 학습한다.

정방향 과정을 다음과 같은 Markov chain으로 정의한다.

```math
q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)
=
\prod_{t=1}^{T}
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})
```

여기서 Markov라는 말은 현재 상태 $\mathbf{x}_{t-1}$만 알면 다음 상태 $\mathbf{x}_t$를 정하는 데 더 이전의 상태가 필요하지 않다는 뜻이다.

```math
q(\mathbf{x}_t
\mid
\mathbf{x}_{t-1},\mathbf{x}_{t-2},\ldots,\mathbf{x}_0)
=
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})
```

아이디어는 강력했지만 당시에는 생성 품질과 계산 효율 면에서 GAN보다 크게 주목받지 못했다.

---

# 4. Score Matching이라는 또 다른 흐름

디퓨전의 발전을 이해하려면 score라는 개념이 필요하다.

확률분포 $p(\mathbf{x})$의 score는 다음과 같다.

```math
\boxed{
\nabla_{\mathbf{x}}
\log p(\mathbf{x})
}
```

이 식은 확률밀도의 로그를 데이터 $\mathbf{x}$에 대해 미분한 벡터다.

## 4.1 score의 기하학적 의미

```math
\nabla_{\mathbf{x}}log p(\mathbf{x})
```

는 현재 위치에서 데이터 밀도가 가장 빠르게 증가하는 방향을 가리킨다.

2차원 공간에 데이터가 구름처럼 모여 있다고 하자.

- 데이터가 많이 존재하는 중심 부근에서는 확률밀도가 높다.
- 데이터가 거의 없는 바깥쪽에서는 확률밀도가 낮다.
- 바깥쪽 위치에서 score 벡터는 대체로 데이터가 밀집한 방향을 가리킨다.

즉 score를 알면 임의의 노이즈 점을 데이터가 많은 영역으로 이동시킬 수 있다.

로그를 사용하는 이유는 다음 관계 때문이다.

```math
\nabla_{\mathbf{x}}\log p(\mathbf{x})
=
\frac{1}{p(\mathbf{x})}
\nabla_{\mathbf{x}}p(\mathbf{x})
```

확률분포의 정규화 상수를 몰라도 기울기 방향을 학습할 수 있다는 장점이 있다.

## 4.2 가우시안 분포의 score

평균이 $\boldsymbol{\mu}$, 공분산이 $\sigma^2\mathbf{I}_d$인 가우시안 분포를 생각하자.

```math
p(\mathbf{x})
=
\mathcal{N}
(\boldsymbol{\mu},\sigma^2\mathbf{I}_d)
```

이 분포의 score는

```math
\boxed{
\nabla_{\mathbf{x}}log p(\mathbf{x})
=
-\frac{\mathbf{x}-\boldsymbol{\mu}}{\sigma^2}
}
```

이다.

$\mathbf{x}$가 평균의 오른쪽에 있으면 score는 왼쪽을 가리키고, 평균의 왼쪽에 있으면 오른쪽을 가리킨다. 즉 항상 확률밀도가 높은 평균 쪽으로 향한다.

## 4.3 Score Matching과 Denoising Score Matching

원래의 Score Matching은 모델 score

```math
\mathbf{s}_\theta(\mathbf{x})
```

가 실제 score

```math
\nabla_{\mathbf{x}}\log p_{\mathrm{data}}(\mathbf{x})
```

를 근사하도록 학습한다.

하지만 실제 데이터 분포의 score는 직접 알 수 없다. Denoising Score Matching은 원본 데이터에 알려진 가우시안 노이즈를 추가하고, 그 조건부 분포의 score를 목표로 사용한다.

노이즈가 섞인 데이터를

```math
\tilde{\mathbf{x}}
=
\mathbf{x}_0
+
\sigma\boldsymbol{\varepsilon}
```

라고 하자.

```math
\boldsymbol{\varepsilon}
\sim
\mathcal{N}(\mathbf{0},\mathbf{I}_d)
```

조건부 분포는

```math
q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x}_0)
=
\mathcal{N}
(\mathbf{x}_0,\sigma^2\mathbf{I}_d)
```

이고 그 score는

```math
\nabla_{\tilde{\mathbf{x}}}
\log q_\sigma
(\tilde{\mathbf{x}}\mid\mathbf{x}_0)
=
-\frac{\tilde{\mathbf{x}}-\mathbf{x}_0}{\sigma^2}
=
-\frac{\boldsymbol{\varepsilon}}{\sigma}
```

이다.

따라서 학습 목표를 직접 계산할 수 있다.

여러 노이즈 크기를 사용하는 [Noise Conditional Score Networks](https://arxiv.org/abs/1907.05600)는 큰 노이즈에서 전체 구조를 잡고 작은 노이즈에서 세부 구조를 복원하는 방식을 발전시켰다.

---

# 5. 2020년: DDPM의 등장

2020년 [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 즉 DDPM은 디퓨전 모델을 크게 대중화했다.

DDPM의 중요한 기여는 다음과 같다.

- 정방향 과정을 단순한 가우시안 전이로 정의했다.
- 임의의 시점 $t$의 noisy data를 한 번에 만들 수 있는 닫힌 형태의 식을 사용했다.
- 복잡한 역과정 평균을 직접 예측하는 대신, 추가된 노이즈를 예측하도록 학습했다.
- 최종 학습식을 단순한 MSE로 만들었다.
- U-Net을 이용해 높은 이미지 생성 품질을 보였다.

DDPM의 핵심은 매우 간단하게 말하면 다음과 같다.

> 원본에 어떤 노이즈를 넣었는지 모델이 맞히게 한다. 생성할 때는 모델이 예측한 노이즈를 반복해서 제거한다.

이제 DDPM의 수식을 순서대로 살펴보자.

---

# 6. DDPM의 정방향 과정

정방향 한 단계는 다음과 같이 정의한다.

```math
\boxed{
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})
=
\mathcal{N}
\left(
\sqrt{\alpha_t}\mathbf{x}_{t-1},
\beta_t\mathbf{I}_d
\right)
}
```

여기서

```math
\alpha_t=1-\beta_t
```

이다.

같은 식을 샘플링 형태로 쓰면

```math
\boxed{
\mathbf{x}_t
=
\sqrt{\alpha_t}\mathbf{x}_{t-1}
+
\sqrt{\beta_t}\boldsymbol{\varepsilon}
}
```

이다.

```math
\boldsymbol{\varepsilon}
\sim
\mathcal{N}(\mathbf{0},\mathbf{I}_d)
```

## 6.1 왜 기존 데이터에 $\sqrt{\alpha_t}$를 곱할까?

노이즈만 계속 더하면 전체 분산이 계속 커진다. DDPM은 기존 신호를 조금 줄이고 그만큼 노이즈를 넣는다.

간단히 각 성분의 평균이 0이고 분산이 1이라고 가정하자.

```math
\mathrm{Var}(\mathbf{x}_{t-1})=1
```

독립인 표준 가우시안 노이즈의 분산도 1이므로

```math
\mathrm{Var}
\left(
\sqrt{\alpha_t}\mathbf{x}_{t-1}
+
\sqrt{\beta_t}\boldsymbol{\varepsilon}
\right)
=
\alpha_t+\beta_t
```

이다.

그런데

```math
\alpha_t+
\beta_t
=
1
```

이므로 전체 분산 규모가 대략 유지된다.

## 6.2 $\beta_t$의 정확한 의미

$\beta_t$는 노이즈의 표준편차가 아니라 **분산**이다.

- 노이즈 분산: $\beta_t$
- 실제 노이즈에 곱하는 표준편차: $\sqrt{\beta_t}$

$\beta_t$는 보통 매우 작은 양수이며 단계에 따라 증가하도록 정한다. 이 일련의 값을 noise schedule이라고 한다.

대표적인 schedule에는 다음이 있다.

- linear schedule
- cosine schedule
- 학습 또는 데이터 특성에 맞게 설계한 schedule

정방향 과정의 $\beta_t$는 모델이 학습하는 값이 아니라 사람이 미리 정하는 값인 경우가 일반적이다.

---

# 7. 임의의 시점으로 한 번에 이동하기

정방향 과정을 실제로 $t$번 반복하지 않아도 $\mathbf{x}_0$에서 $\mathbf{x}_t$를 한 번에 만들 수 있다.

먼저 누적 곱을 정의한다.

```math
\boxed{
\bar{\alpha}_t
=
\prod_{s=1}^{t}\alpha_s
}
```

여기서 막대가 붙은 $\bar{\alpha}_t$는 평균이 아니라 $\alpha_s$들의 누적 곱이다.

그러면 다음이 성립한다.

```math
\boxed{
q(\mathbf{x}_t\mid\mathbf{x}_0)
=
\mathcal{N}
\left(
\sqrt{\bar{\alpha}_t}\mathbf{x}_0,
(1-\bar{\alpha}_t)\mathbf{I}_d
\right)
}
```

샘플링 형태로 쓰면

```math
\boxed{
\mathbf{x}_t
=
\sqrt{\bar{\alpha}_t}\mathbf{x}_0
+
\sqrt{1-\bar{\alpha}_t}
\boldsymbol{\varepsilon}
}
```

이다.

## 7.1 이 식의 의미

```math
\sqrt{\bar{\alpha}_t}\mathbf{x}_0
```

는 아직 남아 있는 원본 신호이고,

```math
\sqrt{1-\bar{\alpha}_t}
\boldsymbol{\varepsilon}
```

는 누적된 노이즈다.

초기에는

```math
\bar{\alpha}_t\approx1
```

이므로 원본 비중이 크다.

후기에는

```math
\bar{\alpha}_t\approx0
```

이므로 원본은 거의 사라지고 노이즈가 대부분을 차지한다.

## 7.2 간단한 유도

두 단계만 전개해 보자.

```math
\mathbf{x}_1
=
\sqrt{\alpha_1}\mathbf{x}_0
+
\sqrt{1-\alpha_1}\boldsymbol{\varepsilon}_1
```

```math
\mathbf{x}_2
=
\sqrt{\alpha_2}\mathbf{x}_1
+
\sqrt{1-\alpha_2}\boldsymbol{\varepsilon}_2
```

첫 번째 식을 두 번째 식에 대입하면

```math
\mathbf{x}_2
=
\sqrt{\alpha_1\alpha_2}\mathbf{x}_0
+
\sqrt{\alpha_2(1-\alpha_1)}
\boldsymbol{\varepsilon}_1
+
\sqrt{1-\alpha_2}
\boldsymbol{\varepsilon}_2
```

이다.

독립인 가우시안 노이즈들의 선형 결합은 다시 가우시안이므로 두 노이즈 항을 하나의 표준 가우시안 노이즈로 합칠 수 있다. 원본 앞의 계수는

```math
\sqrt{\alpha_1\alpha_2}
=
\sqrt{\bar{\alpha}_2}
```

이고, 합쳐진 노이즈의 분산은

```math
1-\alpha_1\alpha_2
=
1-\bar{\alpha}_2
```

가 된다. 이를 일반화하면 위의 닫힌 형태를 얻는다.

## 7.3 학습 속도에 중요한 이유

학습할 때 $1$단계부터 $t$단계까지 순차적으로 노이즈를 넣을 필요가 없다.

1. 원본 $\mathbf{x}_0$를 뽑는다.
2. 임의의 $t$를 뽑는다.
3. 표준 가우시안 노이즈 $\boldsymbol{\varepsilon}$를 뽑는다.
4. 닫힌 형태의 식으로 $\mathbf{x}_t$를 바로 만든다.

이 때문에 한 번의 학습 iteration에서는 임의의 노이즈 수준 하나만 학습해도 된다.

---

# 8. 역방향 과정은 왜 어려운가?

정방향 과정은 사람이 알고 있다.

```math
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})
```

하지만 생성에 필요한 것은 반대 방향이다.

```math
q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)
```

현재의 noisy data만 보고 이전의 덜 noisy한 상태를 정확히 결정할 수는 없다. 같은 $\mathbf{x}_t$가 여러 원본에서 나왔을 수 있기 때문이다.

원본 $\mathbf{x}_0$까지 알고 있다면 실제 후방 조건부분포를 계산할 수 있다.

```math
q(\mathbf{x}_{t-1}
\mid
\mathbf{x}_t,\mathbf{x}_0)
```

가우시안 분포들의 성질을 이용하면 이 분포 역시 가우시안이다.

```math
\boxed{
q(\mathbf{x}_{t-1}
\mid
\mathbf{x}_t,\mathbf{x}_0)
=
\mathcal{N}
\left(
\tilde{\boldsymbol{\mu}}_t
(\mathbf{x}_t,\mathbf{x}_0),
\tilde{\beta}_t\mathbf{I}_d
\right)
}
```

분산은

```math
\boxed{
\tilde{\beta}_t
=
\frac{1-\bar{\alpha}_{t-1}}
{1-\bar{\alpha}_t}
\beta_t
}
```

이고 평균은

```math
\boxed{
\tilde{\boldsymbol{\mu}}_t
(\mathbf{x}_t,\mathbf{x}_0)
=
\frac{
\sqrt{\bar{\alpha}_{t-1}}\beta_t
}{
1-\bar{\alpha}_t
}
\mathbf{x}_0
+
\frac{
\sqrt{\alpha_t}
(1-\bar{\alpha}_{t-1})
}{
1-\bar{\alpha}_t
}
\mathbf{x}_t
}
```

이다.

그런데 생성할 때는 $\mathbf{x}_0$를 모른다. 우리가 알고 있는 것은 현재 상태 $\mathbf{x}_t$뿐이다. 따라서 신경망으로 이 역방향 분포를 근사한다.

```math
p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)
\approx
q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)
```

---

# 9. 왜 원본 대신 노이즈를 예측할까?

정방향 닫힌 형태를 다시 보자.

```math
\mathbf{x}_t
=
\sqrt{\bar{\alpha}_t}\mathbf{x}_0
+
\sqrt{1-\bar{\alpha}_t}
\boldsymbol{\varepsilon}
```

이 식을 $\mathbf{x}_0$에 대해 정리하면

```math
\boxed{
\mathbf{x}_0
=
\frac{
\mathbf{x}_t
-
\sqrt{1-\bar{\alpha}_t}
\boldsymbol{\varepsilon}
}{
\sqrt{\bar{\alpha}_t}
}
}
```

이다.

즉 $\mathbf{x}_t$와 그 안에 들어간 노이즈 $\boldsymbol{\varepsilon}$를 알면 원본을 계산할 수 있다.

DDPM은 신경망이 다음을 예측하도록 한다.

```math
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)
\approx
\boldsymbol{\varepsilon}
```

여기서 신경망 입력에 $t$를 반드시 넣어야 한다. 같은 모양의 입력이라도 현재 노이즈 수준에 따라 제거해야 할 노이즈의 크기와 해석이 달라지기 때문이다.

모델의 예측으로 원본을 추정하면

```math
\boxed{
\hat{\mathbf{x}}_{0,\theta}
=
\frac{
\mathbf{x}_t
-
\sqrt{1-\bar{\alpha}_t}
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)
}{
\sqrt{\bar{\alpha}_t}
}
}
```

이 된다.

노이즈 예측이 사용된 이유는 다음과 같다.

- 학습 목표인 $\boldsymbol{\varepsilon}$를 우리가 정확히 알고 있다.
- 모든 시점에서 목표가 표준 가우시안 노이즈라는 비교적 일관된 형태를 가진다.
- 가우시안 역과정의 평균을 예측하는 문제와 수학적으로 연결된다.
- 실험적으로 안정적이고 단순한 MSE 손실로 좋은 성능을 얻었다.

---

# 10. DDPM의 학습 손실

학습 과정은 다음과 같다.

1. 실제 데이터에서 원본을 뽑는다.

```math
\mathbf{x}_0
\sim
p_{\mathrm{data}}(\mathbf{x})
```

2. 이산 시점 $t$를 균등하게 뽑는다.

```math
t
\sim
\mathrm{Uniform}\{1,\ldots,T\}
```

3. 표준 가우시안 노이즈를 뽑는다.

```math
\boldsymbol{\varepsilon}
\sim
\mathcal{N}(\mathbf{0},\mathbf{I}_d)
```

4. noisy data를 바로 만든다.

```math
\mathbf{x}_t
=
\sqrt{\bar{\alpha}_t}\mathbf{x}_0
+
\sqrt{1-\bar{\alpha}_t}
\boldsymbol{\varepsilon}
```

5. 모델이 노이즈를 예측한다.

```math
\hat{\boldsymbol{\varepsilon}}
=
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)
```

6. 실제 노이즈와 예측 노이즈의 차이를 줄인다.

```math
\boxed{
\mathcal{L}_{\mathrm{simple}}
=
\mathbb{E}_{
\mathbf{x}_0,t,\boldsymbol{\varepsilon}
}
\left[
\left\|
\boldsymbol{\varepsilon}
-
\boldsymbol{\varepsilon}_\theta
\left(
\sqrt{\bar{\alpha}_t}\mathbf{x}_0
+
\sqrt{1-\bar{\alpha}_t}
\boldsymbol{\varepsilon},
t
\right)
\right\|_2^2
\right]
}
```

보통 이를 간단히 다음처럼 쓴다.

```math
\boxed{
\mathcal{L}_{\mathrm{simple}}
=
\mathbb{E}
\left[
\left\|
\boldsymbol{\varepsilon}
-
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)
\right\|_2^2
\right]
}
```

여기서 $\mathbb{E}$는 여러 데이터, 여러 시점, 여러 랜덤 노이즈에 대해 손실을 평균낸다는 뜻이다.

한 번의 학습 iteration에서 전체 $T$단계를 모두 거치는 것이 아니다. 임의의 $t$ 하나를 선택해 그 노이즈 수준에서의 예측 문제 하나를 학습한다. 반복 학습 과정에서 다양한 $t$가 선택되므로 하나의 신경망이 모든 노이즈 수준을 학습하게 된다.

---

# 11. DDPM의 노이즈 예측과 score의 관계

DDPM과 Score-Based Model은 겉으로 보면 서로 다른 값을 예측한다.

- DDPM: 노이즈 $\boldsymbol{\varepsilon}$ 예측
- Score Model: $\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)$ 예측

하지만 두 값은 밀접하게 연결된다.

원본 $\mathbf{x}_0$가 주어졌을 때 noisy data의 조건부분포는

```math
q(\mathbf{x}_t\mid\mathbf{x}_0)
=
\mathcal{N}
\left(
\sqrt{\bar{\alpha}_t}\mathbf{x}_0,
(1-\bar{\alpha}_t)\mathbf{I}_d
\right)
```

이다.

가우시안 score 공식을 적용하면

```math
\nabla_{\mathbf{x}_t}
\log q(\mathbf{x}_t\mid\mathbf{x}_0)
=
-
\frac{
\mathbf{x}_t
-
\sqrt{\bar{\alpha}_t}\mathbf{x}_0
}{
1-\bar{\alpha}_t
}
```

이다.

정방향 식에서

```math
\mathbf{x}_t
-
\sqrt{\bar{\alpha}_t}\mathbf{x}_0
=
\sqrt{1-\bar{\alpha}_t}
\boldsymbol{\varepsilon}
```

이므로

```math
\boxed{
\nabla_{\mathbf{x}_t}
\log q(\mathbf{x}_t\mid\mathbf{x}_0)
=
-
\frac{
\boldsymbol{\varepsilon}
}{
\sqrt{1-\bar{\alpha}_t}
}
}
```

이다.

따라서 모델 score는 노이즈 예측기로 다음처럼 표현할 수 있다.

```math
\boxed{
\mathbf{s}_\theta(\mathbf{x}_t,t)
=
-
\frac{
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)
}{
\sqrt{1-\bar{\alpha}_t}
}
}
```

즉 노이즈를 정확히 예측하는 것은 적절한 비례변환을 거쳐 score를 예측하는 것과 같다.

이 관계 때문에 DDPM과 Score-Based Model은 서로 완전히 독립적인 이론이 아니다. 이후 Score-SDE가 두 관점을 연속시간 틀에서 통합한다.

---

# 12. DDPM의 역방향 생성 과정

모델이 근사하는 역방향 분포를 다음과 같이 둔다.

```math
p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)
=
\mathcal{N}
\left(
\boldsymbol{\mu}_\theta(\mathbf{x}_t,t),
\sigma_t^2\mathbf{I}_d
\right)
```

- $\boldsymbol{\mu}_\theta$: 모델이 결정하는 평균
- $\sigma_t^2$: 역과정의 분산

노이즈 예측을 이용한 평균은 다음과 같다.

```math
\boxed{
\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
\mathbf{x}_t
-
\frac{\beta_t}
{\sqrt{1-\bar{\alpha}_t}}
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)
\right)
}
```

따라서 실제 한 단계 샘플링은

```math
\boxed{
\mathbf{x}_{t-1}
=
\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)
+
\sigma_t\boldsymbol{\xi}_t
}
```

이다.

```math
\boldsymbol{\xi}_t
\sim\mathcal{N}(\mathbf{0},\mathbf{I}_d)
```

마지막 $t=1$ 단계에서는 보통 추가 랜덤 노이즈를 넣지 않는다.

---

## 12.1 역방향 평균식의 직관

```math
\mathbf{x}_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)
```

부분은 모델이 예측한 노이즈 방향을 빼는 것이다.

앞의

```math
\frac{1}{\sqrt{\alpha_t}}
```

는 정방향에서 줄어든 신호 크기를 복원한다.

따라서 역방향 한 단계는 대략

> 예측한 노이즈를 조금 제거하고, 남아 있는 데이터 신호를 다시 키운다.

라고 이해하면 된다.

---

## 12.2 전체 생성 알고리즘

1. 완전한 노이즈를 생성한다.

```math
\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I}_d)
```

2. $t=T,T-1,\ldots,1$ 순서로 반복한다.

```math
\boldsymbol{\varepsilon}_\theta
=
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)
```

```math
\boldsymbol{\mu}_\theta
=
\frac{1}{\sqrt{\alpha_t}}
\left(
\mathbf{x}_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\boldsymbol{\varepsilon}_\theta
\right)
```

```math
\mathbf{x}_{t-1}
=
\boldsymbol{\mu}_\theta
+
\sigma_t\boldsymbol{\xi}_t
```

3. 최종 $\mathbf{x}_0$를 출력한다.

학습 시에는 랜덤한 $t$ 하나만 뽑기 때문에 빠르지만, 생성 시에는 여러 단계를 순차적으로 수행해야 한다. 이것이 디퓨전 모델의 대표적인 속도 문제다.

---

# 13. 원래의 확률적 목적함수: ELBO

DDPM은 원래 최대우도 학습 관점에서 유도된다.

목표는 데이터의 음의 로그우도

```math
-\log p_\theta(\mathbf{x}_0)
```

를 작게 만드는 것이다. 직접 계산하기 어려워 다음 변분 상한을 사용한다.

```math
-\log p_\theta(\mathbf{x}_0)
\leq
\mathcal{L}_{\mathrm{VLB}}
```

개략적인 형태는 다음과 같다.

```math
\begin{aligned}
\mathcal{L}_{\mathrm{VLB}}
=
\mathbb{E}_q\Bigg[
&
D_{\mathrm{KL}}
\left(
q(\mathbf{x}_T\mid\mathbf{x}_0)
\|
p(\mathbf{x}_T)
\right)
\\
&+
\sum_{t=2}^{T}
D_{\mathrm{KL}}
\left(
q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)
\|
p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)
\right)
\\
&-
\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)
\Bigg]
\end{aligned}
```

여기서 KL divergence는 두 확률분포가 얼마나 다른지 측정한다.

```math
D_{\mathrm{KL}}(P\|Q)
=
\mathbb{E}_{x\sim P}
\left[
\log\frac{P(x)}{Q(x)}
\right]
```

중간의 KL 항은

> 실제 역방향 분포와 모델이 예측한 역방향 분포를 비슷하게 만들어라.

라는 뜻이다.

두 분포가 가우시안이고 분산을 고정하면, 평균 차이를 줄이는 문제로 바뀐다. 이를 노이즈 예측 형태로 다시 정리하면 가중된 MSE가 된다. DDPM에서는 실험적으로 더 단순한 비가중 노이즈 MSE를 사용해 좋은 결과를 얻었다.

따라서

```math
\mathcal{L}_{\mathrm{VLB}}
\quad\longrightarrow\quad
\mathcal{L}_{\mathrm{simple}}
```

은 근거 없는 변경이 아니라, 가우시안 역과정에서 유도되는 단순화다.

---

# 14. 왜 한 번에 복원하지 않고 여러 번 복원할까?

완전한 노이즈에서 한 번에 이미지를 예측하면 가능한 원본이 너무 많다.

예를 들어 모델이 “왼쪽으로 가는 행동”과 “오른쪽으로 가는 행동”을 모두 보았다고 하자. 단순 MSE 회귀는 두 행동의 평균인 “가운데로 가는 행동”을 출력할 수 있다. 하지만 가운데에는 장애물이 있을 수도 있다.

디퓨전은 어려운 문제를 작은 문제로 나눈다.

```math
\text{완전한 노이즈}
\rightarrow
\text{아주 noisy한 구조}
\rightarrow
\text{거친 구조}
\rightarrow
\text{세부 구조}
\rightarrow
\text{데이터}
```

각 단계에서는 이전 상태보다 노이즈가 아주 조금 적은 상태만 예측하면 된다. 그래서 여러 모드가 있는 복잡한 분포도 비교적 안정적으로 학습할 수 있다.

---

# 15. 신경망은 어떤 구조인가?

초기 DDPM에서는 주로 U-Net을 사용했다.

입력은

```math
(\mathbf{x}_t,t,\mathbf{c})
```

이다.

- $\mathbf{x}_t$: noisy image
- $t$: 현재 노이즈 수준
- $\mathbf{c}$: 클래스나 텍스트 같은 조건

출력은 입력과 같은 크기의 노이즈다.

```math
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t,\mathbf{c})
```

하나의 신경망이 모든 $t$에 사용된다. 단계마다 별도의 모델을 학습하는 것이 아니다.

$t$는 일반적으로 sinusoidal embedding이나 학습 가능한 embedding으로 변환되어 네트워크에 전달된다. U-Net은 낮은 해상도에서 전체 구조를 파악하고, 높은 해상도에서 세부 정보를 복원한다.

---

# 16. 2020년: DDIM

DDPM은 보통 수백에서 천 단계가 필요했다. DDIM은 **DDPM과 같은 모델을 학습하면서 다른 샘플링 경로를 사용**했다. [DDIM](https://arxiv.org/abs/2010.02502)

먼저 모델의 노이즈 예측으로 원본을 추정한다.

```math
\boxed{
\hat{\mathbf{x}}_{0,\theta}
=
\frac{
\mathbf{x}_t
-
\sqrt{1-\bar{\alpha}_t}
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)
}{
\sqrt{\bar{\alpha}_t}
}
}
```

그다음 이전 상태를 다음처럼 만든다.

```math
\mathbf{x}_{t-1}
=
\sqrt{\bar{\alpha}_{t-1}}
\hat{\mathbf{x}}_{0,\theta}
+
\sqrt{
1-\bar{\alpha}_{t-1}
-
(\sigma_t^{\mathrm{DDIM}})^2
}
\boldsymbol{\varepsilon}_\theta
+
\sigma_t^{\mathrm{DDIM}}\boldsymbol{\xi}_t
```

DDIM의 랜덤성 크기는 보통 다음과 같이 설정한다.

```math
\sigma_t^{\mathrm{DDIM}}
=
\eta
\sqrt{
\frac{1-\bar{\alpha}_{t-1}}
{1-\bar{\alpha}_t}
\left(
1-
\frac{\bar{\alpha}_t}
{\bar{\alpha}_{t-1}}
\right)
}
```

여기서 $\eta$는 DDIM의 랜덤성 조절값으로만 사용한다.

- $\eta=0$: 완전히 결정적인 생성
- $\eta>0$: 확률적인 생성

DDIM의 핵심은 다음과 같다.

- 학습 방법은 DDPM과 동일
- 전체 단계 중 일부만 선택해서 빠르게 생성 가능
- $\eta=0$이면 같은 초기 노이즈에서 항상 같은 결과
- 이미지 interpolation과 inversion에 유리

DDIM은 “새로운 손실함수”라기보다 “새로운 샘플링 방법”에 가깝다.

---

# 17. 2021년: Score-SDE로 통합

DDPM은 이산적인 단계 $t=1,\ldots,T$를 사용한다. Score-SDE는 이를 연속시간 $\tau\in[0,1]$으로 확장했다. [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)

정방향 연속 확산을 다음 확률미분방정식으로 나타낸다.

```math
d\mathbf{r}_\tau
=
\mathbf{f}(\mathbf{r}_\tau,\tau)d\tau
+
g(\tau)d\mathbf{W}_\tau
```

- $\mathbf{r}_0$: 실제 데이터
- $\mathbf{r}_1$: 노이즈
- $\mathbf{f}$: 결정적인 이동 방향
- $g$: 노이즈 크기
- $d\mathbf{W}_\tau$: 아주 작은 가우시안 랜덤 변화

역시간 SDE는 다음과 같다.

```math
d\mathbf{r}_\tau
=
\left[
\mathbf{f}(\mathbf{r}_\tau,\tau)
-
g(\tau)^2
\nabla_{\mathbf{r}_\tau}
\log p_\tau(\mathbf{r}_\tau)
\right]d\tau
+
g(\tau)d\bar{\mathbf{W}}_\tau
```

이 식을 $\tau=1$에서 $\tau=0$ 방향으로 푼다.

중요한 부분은

```math
\nabla_{\mathbf{r}_\tau}\log p_\tau(\mathbf{r}_\tau)
```

즉 score다. score만 알면 노이즈에서 데이터로 되돌아갈 수 있다.

확률적 SDE와 같은 중간 분포를 갖는 결정적 ODE도 존재한다.

```math
\boxed{
\frac{d\mathbf{r}_\tau}{d\tau}
=
\mathbf{f}(\mathbf{r}_\tau,\tau)
-
\frac{1}{2}
g(\tau)^2
\nabla_{\mathbf{r}_\tau}
\log p_\tau(\mathbf{r}_\tau)
}
```

이를 Probability Flow ODE라고 한다.

이 결과가 보여준 것은 다음과 같다.

- DDPM: 이산적인 확률적 역과정
- Score Model: 분포의 score 학습
- DDIM 및 ODE sampler: 결정적 생성 경로

이들이 서로 완전히 별개의 방법이 아니라 같은 연속시간 구조를 다르게 표현한 것이라는 점이다.

---

# 18. 2021~2022년: 조건부 생성과 Guidance

디퓨전 모델이 이미지 생성에서 강력해진 중요한 이유는 조건을 넣기 쉽기 때문이다.

조건부 노이즈 예측기는 다음과 같다.

```math
\boldsymbol{\varepsilon}_\theta
(
\mathbf{x}_t,t,\mathbf{c}
)
```

$\mathbf{c}$는 다음이 될 수 있다.

- 클래스 ID
- 텍스트 임베딩
- 깊이 이미지
- segmentation map
- 로봇 관측
- 목표 상태

---

## 18.1 Classifier Guidance

베이즈 법칙으로

```math
p(\mathbf{x}_t\mid\mathbf{c})
\propto
p(\mathbf{x}_t)p(\mathbf{c}\mid\mathbf{x}_t)
```

이므로 로그를 미분하면

```math
\nabla_{\mathbf{x}_t}
\log p(\mathbf{x}_t\mid\mathbf{c})
=
\nabla_{\mathbf{x}_t}
\log p(\mathbf{x}_t)
+
\nabla_{\mathbf{x}_t}
\log p(\mathbf{c}\mid\mathbf{x}_t)
```

이다.

즉, 기본 디퓨전 score에 분류기가 알려주는 조건 방향을 더한다. 이 방법은 디퓨전 모델이 GAN보다 높은 이미지 품질을 달성하는 데 중요한 역할을 했다. [Diffusion Models Beat GANs](https://arxiv.org/abs/2105.05233)

하지만 noisy image를 분류할 수 있는 별도의 classifier가 필요했다.

---

## 18.2 Classifier-Free Guidance

Classifier-Free Guidance, CFG는 별도 분류기 없이 조건부 예측과 무조건부 예측을 조합한다. [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)

학습 중 일정 확률로 조건을 제거한다.

```math
\mathbf{c}\rightarrow\varnothing
```

그러면 하나의 모델이 다음 두 예측을 모두 할 수 있다.

```math
\boldsymbol{\varepsilon}_{\mathrm{cond}}
=
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t,\mathbf{c})
```

```math
\boldsymbol{\varepsilon}_{\mathrm{uncond}}
=
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t,\varnothing)
```

생성할 때는 다음처럼 결합한다.

```math
\boxed{
\boldsymbol{\varepsilon}_{\mathrm{CFG}}
=
\boldsymbol{\varepsilon}_{\mathrm{uncond}}
+
\gamma
\left(
\boldsymbol{\varepsilon}_{\mathrm{cond}}
-
\boldsymbol{\varepsilon}_{\mathrm{uncond}}
\right)
}
```

여기서 $\gamma$는 guidance scale이다.

- $\gamma=0$: 조건을 무시
- $\gamma=1$: 일반 조건부 예측
- $\gamma>1$: 조건 방향을 더 강하게 강조

$\gamma$를 크게 하면 텍스트 일치도가 높아질 수 있지만 다음 문제가 생길 수 있다.

- 다양성 감소
- 색상 과포화
- 형태 왜곡
- 인공적인 질감

논문이나 구현에 따라 CFG 식을

```math
\boldsymbol{\varepsilon}_{\mathrm{cond}}
+
w(\boldsymbol{\varepsilon}_{\mathrm{cond}}
-\boldsymbol{\varepsilon}_{\mathrm{uncond}})
```

로 쓰기도 한다. 이 경우 $w$의 기준값이 달라지므로 숫자만 비교하면 안 된다.

---

# 19. 2021~2022년: Latent Diffusion

픽셀 공간에서 직접 디퓨전을 하면 계산량이 매우 크다.

예를 들어 $512\times512$ RGB 이미지 전체에 대해 U-Net을 수십 번 또는 수백 번 실행해야 한다.

Latent Diffusion은 먼저 오토인코더로 이미지를 압축한다. [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

```math
\mathbf{z}_0
=
\mathrm{Enc}(\mathbf{x}_0)
```

```math
\hat{\mathbf{x}}_0
=
\mathrm{Dec}(\mathbf{z}_0)
```

- $\mathbf{z}_0$: 압축된 latent representation
- $\mathrm{Enc}$: encoder
- $\mathrm{Dec}$: decoder

이후 픽셀이 아니라 latent에 노이즈를 추가한다.

```math
\mathbf{z}_t
=
\sqrt{\bar{\alpha}_t}\mathbf{z}_0
+
\sqrt{1-\bar{\alpha}_t}
\boldsymbol{\varepsilon}
```

생성 과정은 다음과 같다.

```math
\text{latent noise}
\rightarrow
\text{latent denoising}
\rightarrow
\text{decoder}
\rightarrow
\text{image}
```

텍스트 조건은 일반적으로 cross-attention을 통해 U-Net에 전달된다.

Latent Diffusion의 장점은 다음과 같다.

- 계산량과 메모리 사용량 감소
- 고해상도 생성이 현실적으로 가능
- 텍스트, 이미지 등 다양한 조건을 cross-attention으로 삽입 가능

단점도 있다.

- 오토인코더 압축 과정에서 일부 세부 정보 손실
- 디퓨전 성능뿐 아니라 오토인코더 품질에도 의존
- latent가 사람이 직접 해석할 수 있는 공간이라는 보장은 없음

Stable Diffusion 계열이 이 구조를 대중화했다.

---

# 20. 모델이 예측할 수 있는 세 가지 대상

현대 디퓨전 논문을 읽으면 $\varepsilon$-prediction, $\mathbf{x}_0$-prediction, $v$-prediction이 자주 나온다.

## 20.1 노이즈 예측

```math
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)
\approx
\boldsymbol{\varepsilon}
```

DDPM에서 사용하는 기본 방식이다.

## 20.2 원본 예측

```math
\hat{\mathbf{x}}_{0,\theta}(\mathbf{x}_t,t)
\approx
\mathbf{x}_0
```

노이즈 대신 clean data를 직접 예측한다.

## 20.3 $v$-prediction

다음 값을 예측한다.

```math
\boxed{
\mathbf{v}_t
=
\sqrt{\bar{\alpha}_t}\boldsymbol{\varepsilon}
-
\sqrt{1-\bar{\alpha}_t}\mathbf{x}_0
}
```

$\mathbf{x}_t$와 $\mathbf{v}_t$를 알면 원본과 노이즈를 복원할 수 있다.

```math
\mathbf{x}_0
=
\sqrt{\bar{\alpha}_t}\mathbf{x}_t
-
\sqrt{1-\bar{\alpha}_t}\mathbf{v}_t
```

```math
\boldsymbol{\varepsilon}
=
\sqrt{1-\bar{\alpha}_t}\mathbf{x}_t
+
\sqrt{\bar{\alpha}_t}\mathbf{v}_t
```

$v$-prediction은 노이즈 수준에 따른 학습 불균형을 줄이고 적은 샘플링 단계에서 안정성을 높이는 데 사용된다.

주의할 점은 여기서의 $v$를 Flow Matching의 실제 ODE velocity와 완전히 같은 것으로 생각하면 안 된다는 것이다. 둘 다 velocity라는 용어를 쓰지만 유도되는 방식이 다르다.

---

# 21. 2022년: 샘플링 속도 개선

디퓨전의 주요 약점은 반복적인 신경망 호출이다. 이를 해결하는 방향은 크게 세 가지였다.

## 21.1 더 좋은 수치해석 Solver

Probability Flow ODE를 일반적인 Euler 방법보다 정교하게 푼다.

DPM-Solver는 디퓨전 ODE의 구조를 이용해 약 10~20회 정도의 모델 호출로 좋은 품질을 얻는 방법을 제안했다. [DPM-Solver](https://arxiv.org/abs/2206.00927)

## 21.2 Distillation

많은 단계를 사용하는 teacher 모델을 적은 단계의 student 모델로 압축한다.

Progressive Distillation은 teacher의 두 단계를 student의 한 단계로 반복적으로 압축했다. [Progressive Distillation](https://arxiv.org/abs/2202.00512)

```math
1000\rightarrow500\rightarrow250\rightarrow\cdots\rightarrow4
```

단계처럼 줄여 나간다.

## 21.3 설계 요소 분리

EDM은 다음 요소들이 서로 뒤섞여 있던 문제를 분리해서 분석했다.

- 노이즈 크기 분포
- 데이터와 네트워크 입력의 scaling
- loss weighting
- sampler
- 시간 간격
- stochasticity

EDM은 완전히 다른 생성 모델이라기보다, 디퓨전 모델의 학습 및 샘플링 설계를 체계적으로 정리한 프레임워크에 가깝다. [Elucidating the Design Space of Diffusion Models](https://arxiv.org/abs/2206.00364)

---

# 22. 2022~2023년: U-Net에서 DiT로

초기 디퓨전은 U-Net을 주로 사용했다. DiT는 latent를 이미지 patch처럼 나눈 뒤 Transformer로 처리한다. [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

과정은 대략 다음과 같다.

```math
\mathbf{z}_t
\rightarrow
\text{patch tokens}
\rightarrow
\text{Transformer blocks}
\rightarrow
\text{predicted noise or velocity}
```

시간과 조건은 adaptive layer normalization이나 attention을 통해 전달한다.

중요한 점은 DiT가 새로운 확산 수식을 만든 것은 아니라는 것이다.

```math
\text{DDPM objective}
+
\text{Transformer backbone}
```

에 가깝다.

DiT가 주목받은 이유는 다음과 같다.

- Transformer 크기를 키웠을 때 성능 향상이 비교적 예측 가능
- 텍스트와 이미지 token을 함께 다루기 쉬움
- 대규모 모델 학습에 적합
- 이미지뿐 아니라 비디오와 멀티모달 생성으로 확장하기 쉬움

---

# 23. 2022~2024년: Flow Matching

Flow Matching은 디퓨전의 직접적인 후속으로 자주 소개되지만, 엄밀하게는 별도의 연속형 생성 모델 방법이다. 다만 두 방법은 매우 밀접하다. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

## 23.1 기본 아이디어

Flow Matching에서는 노이즈에서 데이터로 이동하는 연속적인 경로를 직접 만든다.

연속시간은 $\tau\in[0,1]$로 표시하겠다.

- $\tau=0$: 노이즈
- $\tau=1$: 데이터

가장 단순한 선형 경로는

```math
\boxed{
\mathbf{y}_\tau
=
(1-\tau)\boldsymbol{\varepsilon}
+
\tau\mathbf{x}_0
}
```

이다.

$\tau$에 대해 미분하면

```math
\frac{d\mathbf{y}_\tau}{d\tau}
=
\mathbf{x}_0-\boldsymbol{\varepsilon}
```

이 된다.

따라서 목표 속도는

```math
\boxed{
\mathbf{u}^\star
=
\mathbf{x}_0-\boldsymbol{\varepsilon}
}
```

이다.

신경망은 현재 위치에서 어느 방향으로 얼마나 움직여야 하는지를 예측한다.

```math
\mathbf{u}_\theta(\mathbf{y}_\tau,\tau,\mathbf{c})
```

손실함수는

```math
\boxed{
\mathcal{L}_{\mathrm{FM}}
=
\mathbb{E}_{\mathbf{x}_0,\boldsymbol{\varepsilon},\tau}
\left[
\left\|
\mathbf{u}_\theta(\mathbf{y}_\tau,\tau,\mathbf{c})
-
(\mathbf{x}_0-\boldsymbol{\varepsilon})
\right\|_2^2
\right]
}
```

이다.

---

## 23.2 생성 방법

먼저 노이즈에서 시작한다.

```math
\mathbf{y}_0
=
\boldsymbol{\varepsilon}
\sim\mathcal{N}(\mathbf{0},\mathbf{I}_d)
```

그다음 ODE를 푼다.

```math
\boxed{
\frac{d\mathbf{y}_\tau}{d\tau}
=
\mathbf{u}_\theta(\mathbf{y}_\tau,\tau,\mathbf{c})
}
```

가장 단순한 Euler 방법을 쓰면

```math
\mathbf{y}_{\tau+\Delta\tau}
=
\mathbf{y}_\tau
+
\Delta\tau
\mathbf{u}_\theta(\mathbf{y}_\tau,\tau,\mathbf{c})
```

이다.

$\tau=1$에 도착하면 생성된 데이터가 된다.

---

## 23.3 DDPM과 비교

| 항목 | DDPM | 기본 Flow Matching |
|---|---|---|
| 학습 대상 | 노이즈 또는 score | 이동 속도 |
| 경로 | 확률적 diffusion path | 선택한 probability path |
| 생성 방정식 | 역방향 Markov chain/SDE | ODE |
| 랜덤성 | 역과정에 포함 가능 | 주로 초기 노이즈에서 결정 |
| 대표 손실 | 노이즈 MSE | velocity MSE |
| 방향 표현 | 노이즈를 제거할 방향 | 분포를 운반할 방향 |

하지만 두 방법은 완전히 분리되지 않는다.

Flow Matching은 diffusion probability path를 사용할 수도 있다. 반대로 디퓨전 모델도 Probability Flow ODE를 통해 결정적인 ODE로 표현할 수 있다.

따라서 더 정확한 관계는 다음과 같다.

> 디퓨전과 Flow Matching은 모두 단순한 분포와 데이터 분포 사이의 시간 의존적 벡터장을 학습한다. 다만 경로와 학습 대상을 정의하는 방식이 다르다.

---

## 23.4 중요한 미묘한 점

각 학습 쌍의 목표 속도는

```math
\mathbf{x}_0-\boldsymbol{\varepsilon}
```

이지만, 동일한 중간 위치 $\mathbf{y}_\tau$에 여러 경로가 교차할 수 있다.

따라서 MSE의 최적해는 개별 쌍의 속도가 아니라

```math
\mathbf{u}_\theta^\star(\mathbf{y}_\tau,\tau)
=
\mathbb{E}
[
\mathbf{x}_0-\boldsymbol{\varepsilon}
\mid
\mathbf{y}_\tau,\tau
]
```

이다.

그래서 학습 시 선형 interpolation을 사용했다고 해서 실제 생성 trajectory가 모두 완벽한 직선이 되는 것은 아니다.

---

# 24. Rectified Flow

Rectified Flow는 노이즈와 데이터 사이의 경로를 가능한 한 곧고 단순하게 만들어 적은 ODE 단계로 생성하려는 접근이다. [Rectified Flow](https://arxiv.org/abs/2209.03003)

직선 경로라면 Euler 한 단계만으로도 이론적으로 종점에 도달할 수 있다.

```math
\mathbf{y}_1
=
\mathbf{y}_0
+
1\cdot
\mathbf{u}
```

하지만 실제 학습된 velocity field는 경로 교차와 근사 오차 때문에 완벽한 직선이 아니다. 따라서 실제 모델에서는 여전히 여러 단계가 필요할 수 있다.

Rectified Flow에서는 모델이 생성한 noise-data pairing을 다시 사용해 경로를 재학습하는 reflow를 통해 경로를 더 곧게 만들 수 있다.

2024년에는 Rectified Flow와 대규모 multimodal Transformer를 결합한 고해상도 이미지 생성 연구가 등장했다. [Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206)

따라서 최근의 큰 흐름은 대략 다음 조합이다.

```math
\text{latent representation}
+
\text{Transformer}
+
\text{Flow/velocity objective}
+
\text{CFG}
```

다만 “최신 모델은 모두 Flow Matching을 사용한다”고 일반화하면 틀리다. DDPM, score, flow, consistency 계열이 목적에 따라 함께 사용되고 있다.

---

# 25. 2023년 이후: Consistency Model

Consistency Model은 디퓨전의 느린 반복 과정을 한 번 또는 몇 번의 모델 호출로 줄이려는 방법이다. [Consistency Models](https://arxiv.org/abs/2303.01469)

같은 ODE trajectory 위에 있는 두 점을 생각하자.

```math
\mathbf{r}_{\tau_1},
\qquad
\mathbf{r}_{\tau_2}
```

Consistency Model은 두 점을 같은 clean endpoint로 보내도록 학습한다.

```math
\mathbf{F}_\theta(\mathbf{r}_{\tau_1},\tau_1)
\approx
\mathbf{F}_\theta(\mathbf{r}_{\tau_2},\tau_2)
\approx
\mathbf{r}_0
```

그래서 완전한 노이즈에서도 한 번에 데이터를 예측할 수 있다.

```math
\mathbf{r}_0
\approx
\mathbf{F}_\theta(\mathbf{r}_1,1)
```

학습 방법은 크게 두 가지다.

- Consistency distillation: 기존 디퓨전 teacher의 trajectory를 압축
- Consistency training: teacher 없이 consistency 조건을 직접 학습

한 단계 생성은 매우 빠르지만 일반적으로 충분한 반복 단계를 사용하는 diffusion/flow 모델보다 품질과 다양성 유지가 어려울 수 있다.

---

# 26. 로봇 분야에서는 어떻게 사용되는가?

디퓨전은 이미지에만 사용하는 방법이 아니다. 데이터 벡터라면 행동, 오디오, 비디오, 3D 구조에도 적용할 수 있다.

Diffusion Policy에서는 이미지가 아니라 로봇의 행동 시퀀스를 생성한다. [Diffusion Policy](https://arxiv.org/abs/2303.04137)

행동 시퀀스를

```math
\mathbf{a}_0
=
[
\mathbf{a}^{(1)},
\mathbf{a}^{(2)},
\ldots,
\mathbf{a}^{(H)}
]
```

라고 하자.

- $H$: action horizon
- $\mathbf{a}^{(h)}$: $h$번째 시점의 로봇 행동

행동 시퀀스에 노이즈를 넣는다.

```math
\mathbf{a}_t
=
\sqrt{\bar{\alpha}_t}\mathbf{a}_0
+
\sqrt{1-\bar{\alpha}_t}
\boldsymbol{\varepsilon}
```

모델은 로봇 관측 $\mathbf{c}$를 조건으로 노이즈를 예측한다.

```math
\boldsymbol{\varepsilon}_\theta
(
\mathbf{a}_t,t,\mathbf{c}
)
```

여기서 $\mathbf{c}$는 다음과 같다.

- 카메라 영상
- 로봇 관절 상태
- 현재 gripper 상태
- 목표 정보
- 과거 관측

생성 과정은

```math
\text{random action sequence}
\rightarrow
\text{plausible action sequence}
```

가 된다.

이 방법은 같은 관측에서 여러 행동이 가능한 multimodal 상황에 유리하다.

예를 들어 물체를 왼쪽이나 오른쪽으로 피해 갈 수 있다면 단순 MSE policy는 두 경로의 평균을 출력할 수 있다. Diffusion Policy는 두 경로 중 하나를 샘플링할 수 있다.

주의할 점은 여기서 diffusion timestep $t$와 실제 로봇의 행동 시간 $h$가 다르다는 것이다.

- $t$: 노이즈 제거 단계
- $h$: 로봇이 실제로 행동하는 시간 순서

---

# 27. 발전과정을 관통하는 핵심 변화

디퓨전의 발전은 하나의 축이 아니라 여러 축에서 진행됐다.

| 발전 축 | 초기 | 이후 |
|---|---|---|
| 수학적 표현 | 이산 Markov chain | Score-SDE, Probability Flow ODE |
| 예측 대상 | 역과정 평균 | 노이즈, 원본, $v$, velocity |
| 데이터 공간 | 픽셀 공간 | 압축 latent 공간 |
| 네트워크 | U-Net | DiT, multimodal Transformer |
| 조건부 생성 | 클래스 조건 | CFG, text cross-attention |
| 샘플링 | 수백~수천 단계 | DDIM, Solver, Distillation, Consistency |
| 생성 경로 | 확률적 diffusion | 결정적 ODE, Flow Matching |
| 응용 | 이미지 | 비디오, 오디오, 3D, 로봇 행동 |

---

# 28. 자주 생기는 오해

1. 정방향 과정도 학습하는가?

아니다. 정방향 노이즈 과정 $q$는 사람이 미리 정한다. 학습하는 것은 역방향 $p_\theta$다.

2. 학습할 때도 1단계부터 $T$단계까지 모두 실행하는가?

아니다. 랜덤한 $t$ 하나를 선택하고 닫힌 형태의 식으로 $\mathbf{x}_t$를 바로 만든다.

```math
\mathbf{x}_t
=
\sqrt{\bar{\alpha}_t}\mathbf{x}_0
+
\sqrt{1-\bar{\alpha}_t}\boldsymbol{\varepsilon}
```

3. 생성할 때는 왜 반복해야 하는가?

학습한 것은 작은 역방향 이동이므로 이를 여러 번 연결해야 한다. 다만 DDIM, Solver, Distillation 등으로 단계를 줄일 수 있다.

4. 모델이 원본 이미지를 직접 예측하는가?

항상 그렇지는 않다. DDPM은 주로 노이즈를 예측하지만 $\mathbf{x}_0$-prediction이나 $v$-prediction도 가능하다.

5. $\beta_t$가 노이즈 크기인가?

정확히는 노이즈 분산이다. 실제 노이즈에 곱해지는 표준편차는 $\sqrt{\beta_t}$다.

6. $\bar{\alpha}_t$는 평균인가?

아니다. 누적 곱이다.

```math
\bar{\alpha}_t=\prod_{s=1}^t\alpha_s
```

7. 디퓨전과 Flow Matching은 같은가?

같지는 않다. 그러나 둘 다 노이즈 분포와 데이터 분포 사이의 시간 의존적 벡터장을 학습하며, 연속시간 관점에서는 밀접하게 연결된다.

8. Flow Matching이면 무조건 한 단계 생성이 가능한가?

아니다. 학습에 직선 interpolation을 사용해도 실제 학습된 velocity field는 곡선일 수 있다. 수치 오차도 있기 때문에 일반적으로 여러 ODE 단계가 필요하다.

---

# 29. 지금 단계에서 반드시 기억할 식

가장 먼저 기억해야 할 것은 다음 네 개다.

정방향 노이즈 추가:

```math
\boxed{
\mathbf{x}_t
=
\sqrt{\bar{\alpha}_t}\mathbf{x}_0
+
\sqrt{1-\bar{\alpha}_t}\boldsymbol{\varepsilon}
}
```

노이즈 예측 손실:

```math
\boxed{
\mathcal{L}_{\mathrm{simple}}
=
\mathbb{E}
\left[
\left\|
\boldsymbol{\varepsilon}
-
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)
\right\|_2^2
\right]
}
```

노이즈와 score의 관계:

```math
\boxed{
\mathbf{s}_\theta(\mathbf{x}_t,t)
=
-
\frac{
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)
}{
\sqrt{1-\bar{\alpha}_t}
}
}
```

Flow Matching의 기본 식:

```math
\boxed{
\mathbf{y}_\tau
=
(1-\tau)\boldsymbol{\varepsilon}
+
\tau\mathbf{x}_0,
\qquad
\frac{d\mathbf{y}_\tau}{d\tau}
=
\mathbf{x}_0-\boldsymbol{\varepsilon}
}
```

전체 흐름은 다음처럼 정리할 수 있다.

```math
\boxed{
\text{DDPM: 노이즈를 예측해 조금씩 제거}
}
```

```math
\boxed{
\text{Score Model: 데이터 밀도가 증가하는 방향을 예측}
}
```

```math
\boxed{
\text{Flow Matching: 노이즈에서 데이터로 이동하는 속도를 예측}
}
```

이 세 표현은 수학적으로 밀접하게 연결되지만, 학습 대상과 생성 경로가 서로 다르다.
