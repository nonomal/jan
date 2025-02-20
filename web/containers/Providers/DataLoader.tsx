'use client'

import { Fragment, ReactNode, useEffect } from 'react'

import { AppConfiguration } from '@janhq/core'
import { useSetAtom } from 'jotai'

import useAssistants from '@/hooks/useAssistants'
import useGetSystemResources from '@/hooks/useGetSystemResources'
import useModels from '@/hooks/useModels'
import useThreads from '@/hooks/useThreads'

import {
  janDataFolderPathAtom,
  quickAskEnabledAtom,
} from '@/helpers/atoms/AppConfig.atom'

type Props = {
  children: ReactNode
}

const DataLoader: React.FC<Props> = ({ children }) => {
  const setJanDataFolderPath = useSetAtom(janDataFolderPathAtom)
  const setQuickAskEnabled = useSetAtom(quickAskEnabledAtom)

  useModels()
  useThreads()
  useAssistants()
  useGetSystemResources()

  useEffect(() => {
    window.core?.api
      ?.getAppConfigurations()
      ?.then((appConfig: AppConfiguration) => {
        setJanDataFolderPath(appConfig.data_folder)
        setQuickAskEnabled(appConfig.quick_ask)
      })
  }, [setJanDataFolderPath, setQuickAskEnabled])

  console.debug('Load Data...')

  return <Fragment>{children}</Fragment>
}

export default DataLoader
