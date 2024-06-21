# MOMs_WATCH

<img src="./logo.png" width="20%">

영상처리 프로젝트

```bash
python.exe -m pip install --upgrade pip
# GPU 있을 때
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# GPU 없을 때
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```

```bash
Extension
│
├── /.idea    개발환경 설정값이 저장된 폴더입니다.
├── /api      heroku api 설정 폴더입니다.
├── /cam     모델이 웹캠 인풋을 처리하고 익스텐션에 전달하는 폴더입니다.
├── /dist     웹팩 번들링 결과물이 저장된 폴더입니다.
├── /images  UI 아이콘이 저장된 폴더입니다.
├── /page    익스텐션 페이지를 구성하는 파일들이 저장된 폴더입니다.
├── auth.js   로그인과 회원가입을 처리합니다.
├── background.js 페이지들이 상호작용하기 위한 메시지를 전달합니다.
├── manifest.json 크롬 익스텐션에서 필수적인 설정파일입니다.
├── package.json 웹팩 번들링에 필요한 모듈을 정의합니다.
├── package-lock.json 웹팩 번들링에 필요한 모듈을 정의합니다.
├── requirements.txt 모델 개발에 필요한 라이브러리를 정의합니다.
├── webpack.config.cjs 웹팩 번들링의 설정을 정의합니다.
```