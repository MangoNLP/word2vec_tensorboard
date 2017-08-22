# Word2Vec 구현 #  
#### 순서 ####  
```bash
# 라이브러리 설치  
pip -r requirements.txt  

# data.txt.zip 압축 해제

# 실행  
python3 word2vec.py  

# 텐서보드 실행  
tensorboard --logdir=projector  
명령어 치고 브라우저로 localhost:6006 에 접속.  
```  
  
#### 실행 옵션 ####  
- `No options` : 학습 시작(data.txt를 읽어서 학습)
- `-l {keyword}` : 해당 키워드의 가까운 키워드 뽑음


> __Note.__  
학습 시 형태소 분석, 단어 리스트가 꽤 커서 파일로 캐싱시킴.  
새로운 학습이 시작되야 한다면(데이터가 변경된 경우),  
**\*.pk 파일을 삭제**하도록 한다.  
docs_ko.pk : 단어 리스트  
texts_ko.pk : 태깅 리스트  
ko_word2vec.model : 벡터화 파일  
projector 디렉터리 : 텐서보드용 프로젝터 파일  

