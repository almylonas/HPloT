/** @type {import('next').NextConfig} */
const nextConfig = {
  // Move it out of experimental
  serverExternalPackages: ['postgres'],
}

module.exports = nextConfig;
