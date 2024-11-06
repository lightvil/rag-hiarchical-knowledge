# 계층화하여 저장한 지식을 활용한 RAG 답변 품질의 향상

## 서론


### 테스트 방법
* 전처리 및 지식 구축
    1. 오픈 마켓의 모니터에 관한 상품 정보를 텍스트 및 계층화한 Markdown 형식의 문서로 변환
    1. 테스트에 사용할 질문 세트 작성
    1. 변환한 텍스트 문서와 Markdown 형식의 문서를 각각  청킹, 임베딩 과정을 거쳐서 PGVector에 임베딩하여 저장
* 검색 및 답변생성(질문 세트내 개별 질문에 대해)
    1. 텍스트 컬렉션과 Markdown 컬렉션에 대하여 각각 Retriver를 이용하여 답변의 근거가 될 지식들을 검색
        * 텍스트 형식를 저장한 컬렉션은 LangChain의 Retriver를 그대로 사용
        * Markdown 형식을 저장한 컬렉션은 아래에서 설명하는 검색방법을 더하여 사용
    1. 각각 LLM을 통해 답변 생성
* 결과 비교
    * 개별 질문들에 대해 담당자의 답변과 LLM이 생성한 답변들을 함께 비교

### 계층화한 형태로 지식 저장
* 본 프로젝트에서는 계층화한 지식의 형태로 Markdown 형식을 사용한다.
* Markdown 형식이 계층을 표현 형식이면서, 다루기 편한 장점이 있다.
* 오픈 마켓에서 사용자들의 문의가 
### 계층화한 지식의 검색 방법

### 테스트 케이스를 통한 비교

## 결론


## 
### 테스트 환경
* OS: Ubuntu 22.04 LTS
* Anaconda 3
* Python 3.11
* Vector Space: Postgres + PGVector
* Embedding API: OpenAI Embedding
* LLM: ChatGPT-4

### 테스트 프로그램
* `setup/`
    * `create-conda-env`
    * `requirements.txt` : 필요한 패키지들
* `data/`
    * `lg-monitor.md` : 오픈 마켓의 상품 정보를 Markdown 형식으로 변환한것(https://www.11st.co.kr/products/1985839344)
    * `lg-monitor.txt`
    * `questions.json` : 테스트에 사용할 질문들
* `load-text.py`: 
* `load-markdown.py`: 
* `run-test.py` 

### Python 환경 생성
```
# conda virtual environment 생성
conda create --name hiarchical-knowledge python=3.12 

# hiarchical-knowledge 환경을 활성화
conda activate hiarchical-knowledge

# 필요한 패키기 설치
# pip install 
# pip freeze -r setup/requirements.txt

# hiarchical-knowledge 환경을 활성화
pip install -r setup/requirements.txt

```
