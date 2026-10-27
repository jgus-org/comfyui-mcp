{
  description = "MCP server exposing ComfyUI image generation as tools.";

  inputs = {
    # nixpkgs-unstable: fastmcp is not yet in 26.05.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-lib.url = "github:jgus-org/flake-lib/v1";
  };

  outputs = { self, nixpkgs, flake-lib }:
    flake-lib.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        inherit (pkgs) lib;

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          fastmcp
          httpx
          websockets
          pillow
        ]);

        # asyncio sets IPV6_V6ONLY=1 when binding `::`, making the listener IPv6-only; drop that setsockopt so a `::` bind is dual-stack and reachable over IPv4. sitecustomize is auto-imported before server start.
        dualStackHook = pkgs.writeTextDir "sitecustomize.py" ''
          import socket
          _orig = socket.socket.setsockopt
          def setsockopt(self, level, optname, value, *a, **kw):
              if level == socket.IPPROTO_IPV6 and optname == socket.IPV6_V6ONLY and value:
                  return
              return _orig(self, level, optname, value, *a, **kw)
          socket.socket.setsockopt = setsockopt
        '';

        comfyui-mcp = pkgs.stdenv.mkDerivation {
          pname = "comfyui-mcp";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = [ pkgs.makeWrapper ];
          dontConfigure = true;
          dontBuild = true;

          installPhase = ''
            runHook preInstall
            mkdir -p "$out/share/comfyui-mcp" "$out/bin"
            cp -r comfyui_mcp "$out/share/comfyui-mcp/"
            makeWrapper ${pythonEnv}/bin/python "$out/bin/comfyui-mcp" \
              --add-flags "-m comfyui_mcp.server" \
              --chdir "$out/share/comfyui-mcp" \
              --prefix PYTHONPATH : "$out/share/comfyui-mcp" \
              --prefix PYTHONPATH : ${dualStackHook}
            runHook postInstall
          '';

          dontStrip = true;
          meta.mainProgram = "comfyui-mcp";
        };
      in
      {
        packages = {
          inherit comfyui-mcp;
          default = comfyui-mcp;
        };
      });
}
