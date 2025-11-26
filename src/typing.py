from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any, Tuple

class ModelConfig(BaseModel):
    """模型配置模型"""
    model_name: str
    version: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    
class InferenceRequest(BaseModel):
    """推理请求模型"""
    input_data: Union[str, List[str], bytes]
    config: Optional[ModelConfig] = None
    
class InferenceResponse(BaseModel):
    """推理响应模型"""
    success: bool
    output_data: Union[str, List[str], bytes]
    metadata: Optional[Dict[str, Any]] = None
