PORT ?= 8080

.PHONY: setup build build-node build-all test check verify serve clean

setup:
	rustup target add wasm32-unknown-unknown
	cargo install wasm-pack
	npm install

build:
	npm run build

build-node:
	npm run build:node

build-all:
	npm run build:all

test:
	npm run test

check:
	npm run check

verify:
	npm run verify

serve:
	python3 -m http.server $(PORT)

clean:
	rm -rf pkg pkg-node target
