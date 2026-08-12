import { useState, useRef } from 'react'
import { BookOpen, Image as ImageIcon, Download, Sparkles, ChevronRight, Loader2, Feather, ImagePlus, UserCircle, Settings2, PlayCircle, Eye, RefreshCw, Upload, Wand2 } from 'lucide-react'
import { motion, AnimatePresence, useScroll, useTransform } from 'framer-motion'
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

function cn(...inputs) {
  return twMerge(clsx(inputs))
}

const cleanGenres = [
  { id: 'fantasy', label: 'Fantasy' },
  { id: 'scifi', label: 'Sci-Fi' },
  { id: 'mystery', label: 'Mystery' },
  { id: 'romance', label: 'Romance' },
  { id: 'horror', label: 'Horror' },
]

export default function App() {
  const [prompt, setPrompt] = useState('')
  const [genre, setGenre] = useState('Sci-Fi')
  const [pages, setPages] = useState(5)
  const [loading, setLoading] = useState(false)
  const [story, setStory] = useState(null)
  const [generatedImages, setGeneratedImages] = useState({})
  const [generatingImageFor, setGeneratingImageFor] = useState(null)
  const [generatingVideo, setGeneratingVideo] = useState(false)
  const [videoUrl, setVideoUrl] = useState(null)

  const [charLabel, setCharLabel] = useState('')
  const [charImageFile, setCharImageFile] = useState(null)
  const [charImagePreview, setCharImagePreview] = useState(null)

  const containerRef = useRef(null)

  const handleImageUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setCharImageFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setCharImagePreview(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleClearImage = () => {
    setCharImageFile(null)
    setCharImagePreview(null)
  }

  const handleGenerateVideo = async () => {
    if (!story) return
    setGeneratingVideo(true)
    try {
      const response = await fetch('http://localhost:8000/generate/video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          story_data: story,
          image_map: generatedImages
        })
      })
      const data = await response.json()
      setVideoUrl(data.video_url)
    } catch (error) {
      console.error("Error generating video:", error)
    } finally {
      setGeneratingVideo(false)
    }
  }

  const handleGenerateStory = async () => {
    if (!prompt.trim()) return
    setLoading(true)
    setStory(null)
    setGeneratedImages({})
    setVideoUrl(null)

    try {
      const response = await fetch('http://localhost:8000/generate/story', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, genre, pages })
      })
      const data = await response.json()
      await new Promise(r => setTimeout(r, 600)) // smooth transition
      setStory(data)
    } catch (error) {
      console.error("Error generating story:", error)
    } finally {
      setLoading(false)
    }
  }

  const handleGenerateImage = async (pageNum, imagePrompt) => {
    setGeneratingImageFor(pageNum)
    try {
      const formData = new FormData()
      formData.append('prompt', imagePrompt)
      if (charLabel) {
        formData.append('character_label', charLabel)
      }
      if (charImageFile) {
        formData.append('subject_image', charImageFile)
      }

      const response = await fetch('http://localhost:8000/generate/visualize', {
        method: 'POST',
        body: formData
      })
      const data = await response.json()
      setGeneratedImages(prev => ({ ...prev, [pageNum]: data.image_url }))
    } catch (error) {
      console.error("Error generating image:", error)
    } finally {
      setGeneratingImageFor(null)
    }
  }

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.15 }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 30, filter: 'blur(10px)' },
    show: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { type: 'spring', stiffness: 100, damping: 15 } }
  }

  return (
    <div className="min-h-screen bg-[var(--background)] text-slate-900 font-sans selection:bg-slate-200 overflow-x-hidden pt-12 pb-24">
      {/* Very subtle grid background */}
      <div className="fixed inset-0 z-0 pointer-events-none bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>

      <div className="relative z-10 max-w-[1200px] mx-auto px-6 flex flex-col gap-10" ref={containerRef}>

        {/* Minimal Header */}
        <motion.header
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 border-b border-slate-200 pb-8"
        >
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-black rounded-xl flex items-center justify-center flex-shrink-0 shadow-lg">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold tracking-tight font-display text-slate-950">
                LSSN Studio
              </h1>
              <p className="text-slate-500 mt-1 flex items-center gap-2 text-sm tracking-wide">
                Multi-Modal Text & Image Generation Engine
                <span className="px-2 py-0.5 rounded-md text-[10px] font-mono tracking-widest bg-slate-100 border border-slate-200 text-slate-600">
                  LSSN Architecture
                </span>
              </p>
            </div>
          </div>
        </motion.header>

        <main className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">

          {/* Left Panel: Configuration Workbench */}
          <motion.section
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="lg:col-span-4 lg:sticky lg:top-8 space-y-6"
          >
            <div className="bg-white rounded-3xl border border-slate-200 p-8 shadow-xl">
              <div className="flex items-center gap-2 mb-6 text-slate-900">
                <Settings2 className="w-5 h-5" />
                <h2 className="text-lg font-semibold tracking-tight font-display">Generation Settings</h2>
              </div>

              <div className="space-y-6 relative z-10">
                {/* Prompt Input */}
                <div className="space-y-2">
                  <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 ml-1">
                    Core Narrative Prompt
                  </label>
                  <textarea
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl p-4 text-slate-900 placeholder:text-slate-400 focus:ring-2 focus:ring-black focus:border-black outline-none transition-all resize-none text-base leading-relaxed"
                    rows="3"
                    placeholder="Describe your story idea..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                  />
                </div>

                {/* Grid Inputs */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 ml-1">Genre</label>
                    <div className="relative">
                      <select
                        className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 appearance-none outline-none focus:ring-2 focus:ring-black focus:border-black text-slate-900 cursor-pointer transition-colors"
                        value={genre}
                        onChange={(e) => setGenre(e.target.value)}
                      >
                        {cleanGenres.map(g => <option key={g.id} value={g.label}>{g.label}</option>)}
                      </select>
                      <ChevronRight className="w-4 h-4 text-slate-400 absolute right-3 top-1/2 -translate-y-1/2 rotate-90 pointer-events-none" />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 ml-1">Length</label>
                    <div className="relative">
                      <input
                        type="number"
                        className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 pr-16 outline-none focus:ring-2 focus:ring-black focus:border-black text-slate-900 transition-colors"
                        value={pages}
                        onChange={(e) => setPages(Math.min(20, Math.max(1, parseInt(e.target.value) || 1)))}
                        min="1" max="20"
                      />
                      <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] text-slate-400 font-mono tracking-widest">CHAPTERS</span>
                    </div>
                  </div>
                </div>

                {/* Subject consistency */}
                <div className="space-y-4 pt-4 border-t border-slate-100">
                  <div className="flex items-center gap-2 text-slate-800">
                    <UserCircle className="w-4 h-4" />
                    <h3 className="text-sm font-semibold tracking-tight">Subject Reference</h3>
                  </div>

                  <input
                    type="text"
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl py-3 px-4 outline-none focus:ring-2 focus:ring-black focus:border-black text-slate-900 placeholder:text-slate-400 transition-colors text-sm"
                    placeholder="Subject Name (e.g. John Doe)"
                    value={charLabel}
                    onChange={(e) => setCharLabel(e.target.value)}
                  />

                  {!charImagePreview ? (
                    <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-slate-200 border-dashed rounded-xl cursor-pointer hover:bg-slate-50 transition-all group">
                      <Upload className="w-5 h-5 text-slate-400 group-hover:text-slate-600 transition-colors mb-2" />
                      <p className="text-sm text-slate-600 font-medium">Upload Reference</p>
                      <p className="text-[10px] text-slate-400 uppercase tracking-widest mt-1">PNG, JPG</p>
                      <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
                    </label>
                  ) : (
                    <div className="relative w-full h-32 rounded-xl border border-slate-200 overflow-hidden group">
                      <img src={charImagePreview} alt="Reference" className="w-full h-full object-cover" />
                      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center">
                        <button
                          onClick={handleClearImage}
                          className="px-4 py-2 bg-white text-slate-900 text-xs font-semibold rounded-full hover:bg-slate-100 transition-all shadow-sm"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Generate Action */}
                <div className="pt-2">
                  <button
                    onClick={handleGenerateStory}
                    disabled={loading || !prompt}
                    className="w-full py-4 rounded-xl flex items-center justify-center gap-2 font-semibold text-white bg-black hover:bg-slate-800 focus:ring-4 focus:ring-slate-200 transition-all shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Generating Text (Ollama Llama3)...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4" />
                        Run LSSN Generation Pipeline
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </motion.section>

          {/* Right Panel: Story Visualization Feed */}
          <section className="lg:col-span-8 min-h-[800px] relative pb-32">
            <AnimatePresence mode="wait">
              {story ? (
                <motion.div
                  key="story-feed"
                  variants={containerVariants}
                  initial="hidden"
                  animate="show"
                  exit="hidden"
                  className="space-y-8"
                >
                  {/* Story Header */}
                  <motion.div
                    variants={itemVariants}
                    className="flex flex-col md:flex-row items-start md:items-end justify-between pb-6 mb-8 gap-6 border-b border-slate-200"
                  >
                    <div>
                      <h2 className="text-4xl lg:text-5xl font-bold font-display text-slate-950 mb-4 tracking-tight leading-tight">
                        {story.title}
                      </h2>
                      <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-slate-500">
                        <span className="px-2 py-1 rounded bg-slate-100 border border-slate-200 text-slate-700">{story.genre}</span>
                        <span className="flex items-center gap-1"><BookOpen className="w-3 h-3" /> {story.pages.length} Chapters</span>
                      </div>
                    </div>
                    <div className="flex shrink-0 w-full md:w-auto">
                      {videoUrl ? (
                        <a
                          href={videoUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="w-full md:w-auto px-6 py-3 bg-black text-white rounded-xl text-sm font-semibold hover:bg-slate-800 transition-all flex items-center justify-center gap-2 shadow-sm"
                        >
                          <Download className="w-4 h-4" /> Download Cinematic
                        </a>
                      ) : (
                        <button
                          onClick={handleGenerateVideo}
                          disabled={generatingVideo || Object.keys(generatedImages).length === 0}
                          className="w-full md:w-auto px-6 py-3 bg-white border border-slate-200 hover:bg-slate-50 rounded-xl text-slate-900 text-sm font-semibold transition-all shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                          {generatingVideo ? <Loader2 className="w-4 h-4 animate-spin text-slate-400" /> : <PlayCircle className="w-4 h-4 text-slate-400" />}
                          {generatingVideo ? 'Rendering Engine...' : 'Compile Video'}
                        </button>
                      )}
                    </div>
                  </motion.div>

                  {/* Chapters Loop */}
                  {story.pages.map((page, index) => (
                    <motion.div
                      key={page.page_number}
                      variants={itemVariants}
                      className="bg-white border border-slate-200 rounded-3xl overflow-hidden shadow-xl hover:shadow-2xl transition-shadow duration-300"
                    >
                      <div className="grid lg:grid-cols-[1fr,minmax(350px,40%)] gap-0">
                        {/* Narrative Text Side */}
                        <div className="p-8 md:p-10 flex flex-col justify-center border-b lg:border-b-0 lg:border-r border-slate-100 order-2 lg:order-1 relative bg-white">
                          <span className="absolute top-6 left-6 text-[80px] font-display font-bold text-slate-50 select-none pointer-events-none">
                            {page.page_number.toString().padStart(2, '0')}
                          </span>
                          <div className="relative z-10 mt-8">
                            <h3 className="text-xs font-bold tracking-widest uppercase text-slate-400 mb-4 flex items-center gap-2">
                              Chapter {page.page_number}
                            </h3>
                            <p className="text-slate-800 leading-relaxed text-lg font-normal mb-8">
                              {page.text}
                            </p>

                            <div className="pt-6 border-t border-slate-100">
                              <p className="text-[10px] text-slate-400 font-mono mb-2 uppercase tracking-widest flex items-center gap-1">
                                <Eye className="w-3 h-3" /> System Prompt Extract
                              </p>
                              <p className="text-sm text-slate-500 italic bg-slate-50 p-4 rounded-xl border border-slate-100">
                                "{page.image_prompt}"
                              </p>
                            </div>
                          </div>
                        </div>

                        {/* Visual Canvas Side */}
                        <div className="min-h-[350px] lg:min-h-full bg-slate-50 relative flex items-center justify-center p-6 order-1 lg:order-2">
                          {generatedImages[page.page_number] ? (
                            <motion.div
                              initial={{ opacity: 0, scale: 0.95 }}
                              animate={{ opacity: 1, scale: 1 }}
                              className="relative w-full aspect-square rounded-2xl overflow-hidden border border-slate-200 shadow-md group"
                            >
                              <img
                                src={generatedImages[page.page_number]}
                                alt={`Chapter ${page.page_number} visual`}
                                className="w-full h-full object-cover"
                              />
                              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors duration-200 flex items-center justify-center">
                                <button
                                  onClick={() => handleGenerateImage(page.page_number, page.image_prompt)}
                                  className="opacity-0 group-hover:opacity-100 text-xs font-semibold text-slate-900 bg-white shadow-lg px-5 py-2.5 rounded-full border border-slate-200 hover:bg-slate-50 transition-all flex items-center gap-2 transform translate-y-2 group-hover:translate-y-0"
                                >
                                  <RefreshCw className="w-3 h-3" /> Regenerate Visual
                                </button>
                              </div>
                            </motion.div>
                          ) : (
                            <div className="text-center w-full max-w-xs space-y-4">
                              <div className="w-20 h-20 mx-auto rounded-full bg-white flex items-center justify-center border border-slate-200 text-slate-400 shadow-inner">
                                {generatingImageFor === page.page_number ? (
                                  <Loader2 className="w-8 h-8 animate-spin text-slate-600" />
                                ) : (
                                  <ImagePlus className="w-8 h-8" />
                                )}
                              </div>

                              <div>
                                <h4 className="text-slate-900 font-semibold text-sm">Synthesize Visuals</h4>
                                <p className="text-xs text-slate-500 mt-1">Execute the local LSSN UNet pass.</p>
                              </div>

                              <button
                                onClick={() => handleGenerateImage(page.page_number, page.image_prompt)}
                                disabled={generatingImageFor === page.page_number}
                                className="w-full px-4 py-3 bg-white border border-slate-200 hover:bg-slate-50 text-slate-900 text-xs text-center font-semibold rounded-xl transition-all shadow-sm disabled:opacity-50"
                              >
                                {generatingImageFor === page.page_number ? 'Processing Tensor...' : 'Generate Image'}
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </motion.div>
              ) : (
                <motion.div
                  key="empty-state"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.4 }}
                  className="flex flex-col items-center justify-center text-center mt-20 md:mt-0 md:min-h-[500px] border border-slate-200 rounded-3xl bg-white shadow-xl"
                >
                  <div className="w-20 h-20 bg-slate-50 rounded-2xl flex items-center justify-center mb-6 border border-slate-100">
                    <Wand2 className="w-8 h-8 text-slate-300" />
                  </div>
                  <h3 className="text-2xl font-bold font-display text-slate-900 mb-3 tracking-tight">System Ready</h3>
                  <p className="text-slate-500 max-w-sm mx-auto leading-relaxed text-sm">
                    Enter a core narrative prompt and press Generate to establish the local LSSN model inference.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </section>
        </main>
      </div>
    </div>
  )
}
