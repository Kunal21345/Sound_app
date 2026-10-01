import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  experimental: {
    proxyClientMaxBodySize: "26mb",
    proxyTimeout: 600000,
  },
  async rewrites() {
    return [
      { source: "/api/:path*", destination: "http://127.0.0.1:5000/:path*" },
      { source: "/audio/:path*", destination: "http://127.0.0.1:5000/audio/:path*" },
    ]
  },
}

export default nextConfig
