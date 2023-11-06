import { Model } from '@janhq/core/lib/types'
import DownloadModelContent from '../DownloadModelContent'

type Props = {
  model: Model
  isRecommend: boolean
  required?: string
  transferred?: number
  onDeleteClick?: (model: Model) => void
}

const DownloadedModelCard: React.FC<Props> = ({
  model,
  isRecommend,
  required,
  onDeleteClick,
}) => (
  <div className="rounded-lg border border-gray-200">
    <div className="flex justify-between gap-2.5 px-3 py-4">
      <DownloadModelContent
        required={required}
        author={model.author}
        description={model.shortDescription}
        isRecommend={isRecommend}
        name={model.name}
        type={'LLM'}
      />
      <div className="flex flex-col justify-center">
        <button onClick={() => onDeleteClick?.(model)}>Delete</button>
      </div>
    </div>
  </div>
)

export default DownloadedModelCard