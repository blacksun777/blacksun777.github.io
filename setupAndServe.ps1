echo docker run -p 4000:4000 --name bs777site -v '${PWD}':/srv/jekyll jekyll/jekyll jekyll serve
docker run -p 4000:4000 --name bs777site -v ${PWD}:/srv/jekyll jekyll/jekyll jekyll serve