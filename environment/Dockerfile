FROM public.ecr.aws/x8v8d7g8/mars-base:latest
WORKDIR /app
ARG REPO_URL=https://github.com/moov-io/watchman
ARG BASE_SHA=bfdd8a0b0959bfcf62bf2cce618f7a1f9e01d50c
RUN git clone "$REPO_URL" . \
 && DEFAULT="$(git remote show origin | sed -n 's/.*HEAD branch: //p')" \
 && git checkout -B "$DEFAULT" "$BASE_SHA" \
 && git remote remove origin \
 && git reflog expire --expire=now --all \
 && git gc --prune=now
RUN go mod download
CMD ["bash"]
