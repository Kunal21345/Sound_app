import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "Text to speech — Voice studio",
  description: "Shape your words into expressive, natural sounding speech.",
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>
}
