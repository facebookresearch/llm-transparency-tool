FROM python:3.10

RUN apt-get update
RUN apt-get install npm -y
RUN npm install --global yarn

COPY llm_transparency_tool /llm_transparency_tool
COPY requirements_contributions.txt /requirements.txt
COPY setup.py /

# TODO(igortufanov): remove this line when the tool repo forks out
RUN mkdir /transparent_llms

WORKDIR /
RUN pip install -r requirements.txt
RUN pip install -e .

WORKDIR /llm_transparency_tool/components/frontend
RUN yarn install
RUN yarn build

ENV HOME=/llm_transparency_tool/server

EXPOSE 80

ENTRYPOINT ["streamlit", "run", "/llm_transparency_tool/server/app.py", \
    "--server.port=80", "--", "/llm_transparency_tool/config/docker_local.json"]
