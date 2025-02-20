'use client'

import { PropsWithChildren, useEffect, useState } from 'react'

import { Toaster } from 'react-hot-toast'

import { usePathname } from 'next/navigation'

import { TooltipProvider } from '@janhq/uikit'

import GPUDriverPrompt from '@/containers/GPUDriverPromptModal'
import EventListenerWrapper from '@/containers/Providers/EventListener'
import JotaiWrapper from '@/containers/Providers/Jotai'
import ThemeWrapper from '@/containers/Providers/Theme'

import { setupCoreServices } from '@/services/coreService'
import {
  isCoreExtensionInstalled,
  setupBaseExtensions,
} from '@/services/extensionService'

import Umami from '@/utils/umami'

import Loader from '../Loader'

import DataLoader from './DataLoader'

import KeyListener from './KeyListener'

import { extensionManager } from '@/extension'

const Providers = (props: PropsWithChildren) => {
  const { children } = props
  const pathname = usePathname()

  const [setupCore, setSetupCore] = useState(false)
  const [activated, setActivated] = useState(false)
  const [settingUp, setSettingUp] = useState(false)

  async function setupExtensions() {
    // Register all active extensions
    await extensionManager.registerActive()

    setTimeout(async () => {
      if (!isCoreExtensionInstalled()) {
        // TODO: Proper window handle
        // Do not migrate extension from quick ask window
        if (pathname === '/search') {
          return
        }
        setSettingUp(true)
        await setupBaseExtensions()
        return
      }

      extensionManager.load()
      setSettingUp(false)
      setActivated(true)
    }, 500)
  }

  // Services Setup
  useEffect(() => {
    setupCoreServices()
    setSetupCore(true)
    return () => {
      extensionManager.unload()
    }
  }, [])

  useEffect(() => {
    if (setupCore) {
      // Electron
      if (window && window.core?.api) {
        setupExtensions()
      } else {
        // Host
        setActivated(true)
      }
    }
  }, [setupCore])

  return (
    <JotaiWrapper>
      <ThemeWrapper>
        <Umami />
        {settingUp && <Loader description="Preparing Update..." />}
        {setupCore && activated && (
          <KeyListener>
            <EventListenerWrapper>
              <TooltipProvider delayDuration={0}>
                <DataLoader>{children}</DataLoader>
              </TooltipProvider>
              {!isMac && <GPUDriverPrompt />}
            </EventListenerWrapper>
            <Toaster />
          </KeyListener>
        )}
      </ThemeWrapper>
    </JotaiWrapper>
  )
}

export default Providers
