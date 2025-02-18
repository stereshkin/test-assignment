from pydantic import BaseModel, Field


class DocumentationChangeResponse(BaseModel):
    changed_feature: str = Field(description="The exact feature or functionality that was changed, using the exact wording \
                                from the user query. For example, if the query states 'We improved the query editor by adding real-time \
                                syntax highlighting and error detection', you must extract 'query editor' as the changed feature—not a broader \
                                term like 'editor' or 'interface.'")
    changes: str = Field(description="A clear, concise description of how this feature was changed.")


class UpdatedDocument(BaseModel):
    ind: int = Field(description="id of the document")
    content_before: str = Field(description="content of the document provided to you, i.e. before any changes are applied")
    content_after: str = Field(description="content of the document after you applied your changes, i.e. updated content")
    url: str = Field(description="url to this document")


class UpdatedDocumentsResponse(BaseModel):
    updated_documents: list[UpdatedDocument] = Field(description="list of updated documents")

