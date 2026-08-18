import { useCallback, useRef, useState } from 'react'
import { getConfluence, getErrorMessage, getExposure, getVolumeProfile } from '../api/client'
import type { ConfluenceResponse, ExposureResponse, VolumeProfileResponse } from '../types'

export function useConfluenceLevels(symbol: string) {
  const [data, setData] = useState<ConfluenceResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const dataRef = useRef(data)
  dataRef.current = data

  const fetchLevels = useCallback(async () => {
    if (!symbol) return
    try {
      setIsLoading(true)
      setError(null)
      const next = await getConfluence({ symbol })
      if (!next.levels?.length) {
        setError('No confluence levels returned')
        setData(null)
        return
      }
      setData(next)
    } catch (err) {
      setError(getErrorMessage(err))
      setData(null)
    } finally {
      setIsLoading(false)
    }
  }, [symbol])

  const toggle = useCallback(async () => {
    if (!symbol) return
    if (dataRef.current) {
      setData(null)
      setError(null)
      return
    }
    await fetchLevels()
  }, [symbol, fetchLevels])

  const reset = useCallback(() => {
    setData(null)
    setError(null)
  }, [])

  return { data, isLoading, error, toggle, reset }
}

export function useVolumeProfileLevels(symbol: string, timeframe: string) {
  const [data, setData] = useState<VolumeProfileResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const dataRef = useRef(data)
  dataRef.current = data

  const fetchLevels = useCallback(async () => {
    if (!symbol) return
    try {
      setIsLoading(true)
      setError(null)
      const next = await getVolumeProfile({ symbol, timeframe })
      if (next.poc == null && next.vah == null && next.val == null) {
        setError('No volume-profile levels returned')
        setData(null)
        return
      }
      setData(next)
    } catch (err) {
      setError(getErrorMessage(err))
      setData(null)
    } finally {
      setIsLoading(false)
    }
  }, [symbol, timeframe])

  const toggle = useCallback(async () => {
    if (!symbol) return
    if (dataRef.current) {
      setData(null)
      setError(null)
      return
    }
    await fetchLevels()
  }, [symbol, fetchLevels])

  const reset = useCallback(() => {
    setData(null)
    setError(null)
  }, [])

  return { data, isLoading, error, toggle, reset }
}

export function useExposureOverlay(symbol: string) {
  const [data, setData] = useState<ExposureResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const dataRef = useRef(data)
  dataRef.current = data

  const fetchLevels = useCallback(async () => {
    if (!symbol) return
    try {
      setIsLoading(true)
      setError(null)
      setData(await getExposure(symbol))
    } catch (err) {
      setError(getErrorMessage(err))
      setData(null)
    } finally {
      setIsLoading(false)
    }
  }, [symbol])

  const toggle = useCallback(async () => {
    if (!symbol) return
    if (dataRef.current) {
      setData(null)
      setError(null)
      return
    }
    await fetchLevels()
  }, [symbol, fetchLevels])

  const reset = useCallback(() => {
    setData(null)
    setError(null)
  }, [])

  return { data, isLoading, error, toggle, reset }
}
