FROM python:3.11-slim

ARG RUNNER_VERSION=2.331.0

# install deps + kubectl
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git jq unzip \
  && rm -rf /var/lib/apt/lists/*

# install kubectl
RUN ARCH="$(dpkg --print-architecture)" \
 && curl -fsSL -o /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/${ARCH}/kubectl" \
 && chmod +x /usr/local/bin/kubectl

# create runner user
RUN useradd -m -u 1001 -s /bin/bash runner

WORKDIR /actions-runner

# download and extract GitHub runner
RUN curl -fsSL -o actions-runner-linux-x64.tar.gz \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
 && tar xzf actions-runner-linux-x64.tar.gz \
 && rm -f actions-runner-linux-x64.tar.gz \
 && chown -R runner:runner /actions-runner

# install python deps you need often (optional)
RUN pip install --no-cache-dir --upgrade pip

COPY runner/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown runner:runner /entrypoint.sh

USER runner
ENTRYPOINT ["/entrypoint.sh"]